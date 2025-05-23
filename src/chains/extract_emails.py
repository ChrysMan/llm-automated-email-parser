
import asyncio, os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch 
from accelerate import PartialState
from accelerate.utils import gather_object
from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline, BitsAndBytesConfig
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFacePipeline
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Tuple
from time import time
from utils.logging_config import LOGGER
from utils.graph_utils import extract_msg_file, clean_data, split_email_thread, write_file, append_file, chunk_emails



langsmith_api_key = os.environ.get("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
if not langsmith_api_key:
    LOGGER.warning("Langsmith API key not found. Tracing will be disabled.")
    

#os.environ["LANGCHAIN_TRACING"] = "true"

class EmailInfo(BaseModel):
    """Data model for email extracted information."""
    sender: str = Field(..., description="The sender of the email. Store their name and email, if available.")
    sent: str = Field(..., description="The date the email was sent.")
    to: Optional[List[str]] = Field(default_factory=list, description="A list o recipients' names and emails.")
    cc: Optional[List[str]] = Field(default_factory=list, description="A list of additional recipients' names and emails.")
    subject: Optional[str] = Field(None, description="The subject of the email.")
    body: str = Field(..., description="The email body, excluding unnecessary whitespace.")

#model = OllamaLLM(model="llama3.1",temperature=0, num_ctx=32768, num_predict=16384)
parser = JsonOutputParser(pydantic_object=EmailInfo, json_compatible=True)

prompt_template = PromptTemplate.from_template(
"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert assistant who processes email threads with precision and no loss of content.
The input consists of several email strings. Some emails may be duplicates. Your task is:
1. Split the email thread into individual emails using "***" as the only delimiter. Each "***" marks the end of one complete email.
2. Identify and remove any duplicate emails in the list based on content mainting the chronological order.
3. Return the unique emails **ONLY** as a JSON list, where each email is a **separate object** with the following fields:

- sender: The sender of the email. Store their name and email, if available, as a string in the format "Name <email@example.com>". If only a name is present, store it as "Name". 
- sent: Extract and include **only the date and time**. 
- to: A list of recipients' names and emails, if available. Store each entry as a string in the format "Name <email@example.com>" or "Name". The "To" field may contain multiple recipients separated by commas or semicolons but is usually one.
- cc: A list of additional recipients' names and emails. Store each entry as a string in the format "Name <email@example.com>" or "Name". The "Cc" field may contain multiple recipients separated by commas or semicolons.
- subject: The subject of the email, stored as a string. Extract this after the "Subject:" field.
- body:  Include the full content of the message starting from the line **after** "Subject:" or "wrote:", and continue **until the next delimiter**. Do not summarize or skip any content.

4. Maintain the chronological order of the emails in the output.
5. Do not hallucinate or add any information that is not present in the email thread.
6. The length of the JSON list needs to be the same as the number of the emails strictly.
7. Output ONLY the raw JSON array. Do NOT include extra quotes, explanations, or any text.

Process the following email thread:
<|eot_id|><|start_header_id|>user<|end_header_id|>
{emails}
<|eot_id|><|start_header_id|>assistant<|end_header_id>
"""
)

def load_model(use_accelerate: bool = False, model_id: str = "meta-llama/Llama-3.1-8B-Instruct") -> Tuple[AutoModelForCausalLM, AutoTokenizer, Optional[PartialState]]:
    """
    Load the model and tokenizer for email extraction.

    Args:
        use_accelerate (bool): Whether to use the accelerate library for model loading.
        model_id (str): The model ID to load.

    Returns:
        Tuple[AutoTokenizer, AutoModelForCausalLM]: The loaded tokenizer and model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if use_accelerate:
        distributed_state = PartialState()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            #device_map="auto"
            device_map={"": distributed_state.device}  # place model on local GPU
        )

        return model, tokenizer, distributed_state
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="balanced",
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True
            )
    
    return model, tokenizer, None

def load_pipeline(model, tokenizer):
    """
    Load the HuggingFace pipeline for text generation.

    Args:
        model (AutoModelForCausalLM): The loaded model.
        tokenizer (AutoTokenizer): The loaded tokenizer.

    Returns:
        pipeline: The HuggingFace pipeline for text generation.
    """
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens = 256,
        return_full_text=False,
        do_sample=False,
        temperature=None,
        top_k=None,
        top_p=None
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    SPLIT_EMAILS_CHAIN = (prompt_template | llm | parser)

    return SPLIT_EMAILS_CHAIN

async def process_chunk_async(chunk: List, chunk_size, chain: Runnable) -> List[Dict]:
    chunk = "\n*** \n".join(chunk)
    tic = time()
    result = await chain.ainvoke({"emails": chunk}, config={"metadata": {"chunk_size": chunk_size, "invocation_method": "async"}, "run_name": f"async_chunk_{chunk_size}"})
    LOGGER.info(f"Chunk {chunk_size} took {time() - tic:.2f}s")
    return result

def split_and_extract_emails_sync(file_path: str) -> List[Dict]:
    """
    Synchronous version with serial invocations

    Splits a .msg file containing an email thread into individual cleaned emails, extracts metadata (sender, date, etc.),
    and returns them in chronological order (oldest first). 

    Args:
        file_path (str): Path to the .msg file containing the email thread.
    Returns:
        List[Dict]: A list of dictionaries, each containing the extracted email information.
    """ 

    os.environ["LANGCHAIN_PROJECT"] = "extract_emails_sync"

    # Get local rank from environment
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    
    model, tokenizer = load_model()
    split_emails_chain = load_pipeline(model, tokenizer)

    raw_msg_content = extract_msg_file(file_path)
    cleaned_msg_content = clean_data(raw_msg_content)
    write_file(cleaned_msg_content, "clean1.txt")
    LOGGER.info("Splitting emails...")
    try:
        raw_model_output = []
        splitted_emails = split_email_thread(cleaned_msg_content)[::-1]
        chunk_size = 5
        for chunk in chunk_emails(splitted_emails, chunk_size=chunk_size):
            chunk = "\n*** \n".join(chunk)
            append_file(chunk, "emailSplit.txt")

            # Invoke the model with the chunk
            tic = time()
            raw_model_output.extend(split_emails_chain.invoke({"emails": chunk}, config={"metadata": {"chunk_size": chunk_size, "invocation_method": "sync"}, "run_name": f"sync_chunk_{chunk_size}"}))
            LOGGER.info(f"Time taken for chunk: {time() - tic:.2f} seconds")
            
        LOGGER.info(f"Total output emails: {len(raw_model_output)}")        
    
    except Exception as e:
        LOGGER.error(f"Failed to split emails: {e}")

    return raw_model_output

async def split_and_extract_emails_async(file_path: str) -> List[Dict]:
    """
    Asynchronous version with parallel invocations

    Splits a .msg file containing an email thread into individual cleaned emails, extracts metadata (sender, date, etc.),
    and returns them in chronological order (oldest first). 

    Args:
        file_path (str): Path to the .msg file containing the email thread.
    Returns:
        List[Dict]: A list of dictionaries, each containing the extracted email information.
    """ 

    os.environ["LANGCHAIN_PROJECT"] = "extract_emails_async"
    model, tokenizer = load_model()
    split_emails_chain = load_pipeline(model, tokenizer)

    raw_msg_content = extract_msg_file(file_path)
    cleaned_msg_content = clean_data(raw_msg_content)
    write_file(cleaned_msg_content, "clean1.txt")
    LOGGER.info("Splitting emails...")
    try:
        raw_model_output = []
        chunk_size=6
        splitted_emails = split_email_thread(cleaned_msg_content)[::-1]
        chunks = chunk_emails(splitted_emails, chunk_size=chunk_size)
        tasks = [process_chunk_async(chunk, chunk_size, split_emails_chain) for chunk in chunks]
        results = await asyncio.gather(*tasks)

        for result in results:
            raw_model_output.extend(result)           
        
        LOGGER.info(f"Total output emails: {len(results)}")        
    
    except Exception as e:
        LOGGER.error(f"Failed to split emails: {e}")

    return raw_model_output

def process_chunk(chunk: str, chunk_idx, chain: Runnable, distributed_state: PartialState) -> List[Dict]:
    #chunk = "\n*** \n".join(chunk)
    tic = time()
    result = chain.invoke({"emails": chunk}, config={"metadata": {"chunk_num": chunk_idx, "invocation_method": "distributed inference"}, "run_name": f"chunk_{chunk_idx}"})
    LOGGER.info(f"[GPU {distributed_state.process_index}] Chunk {chunk_idx} took {time() - tic:.2f}s")
    return result

def distributed_email_inference(email_chunks: List[str], chain: Runnable, distributed_state: PartialState) -> List[Dict]:
    results = []
    with distributed_state.split_between_processes(email_chunks) as local_chunks:
        # Process each chunk in parallel
        #tasks = [process_chunk_async(chunk, idx, chain) for idx, chunk in enumerate(local_chunks)]
        #results = asyncio.run(asyncio.gather(*tasks))
        
        for idx, chunk in enumerate(local_chunks):
            results.extend(process_chunk(chunk, idx, chain, distributed_state))
    return results

def split_and_extract_emails_acc(file_path: str) -> List[Dict]:
    """
    Distributed version with parallel invocations using accelerate

        Splits a .msg file containing an email thread into individual cleaned emails, extracts metadata (sender, date, etc.),
    and returns them in chronological order (oldest first). 

    Args:
        file_path (str): Path to the .msg file containing the email thread.
    Returns:
        List[Dict]: A list of dictionaries, each containing the extracted email information.
    """ 


    os.environ["LANGCHAIN_PROJECT"] = "extract_emails_acc"
    model, tokenizer, distributed_state = load_model(use_accelerate=True)
    split_emails_chain = load_pipeline(model, tokenizer)

    #accelerator.wait_for_everyone()
    #wait_for_all(distributed_state)

    raw_msg_content = extract_msg_file(file_path)
    cleaned_msg_content = clean_data(raw_msg_content)
    LOGGER.info("Splitting emails...")
    try:
        splitted_emails = split_email_thread(cleaned_msg_content)[::-1]
        chunk_size=5
        #chunks = list(chunk_emails(splitted_emails, chunk_size=chunk_size))
        partial_results = distributed_email_inference(splitted_emails, split_emails_chain, distributed_state)

        #accelerator.wait_for_everyone()
        #wait_for_all(accelerator)

        all_results = gather_object(partial_results)
        
        distributed_state.wait_for_everyone()

        if distributed_state.is_main_process:
            flat_results = [item for sublist in all_results for item in sublist]
            LOGGER.info(f"Total output emails: {len(flat_results)}")
            return flat_results
        else:
            return []
    
    except Exception as e:
        LOGGER.error(f"Failed to split emails: {e}")

    return []

