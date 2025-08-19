
import asyncio, os
import torch 
import math
from langchain_ollama import OllamaLLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFacePipeline
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Tuple
from time import time
from utils.logging_config import LOGGER
from utils.graph_utils import extract_msg_file, clean_data, split_email_thread, chunk_emails,find_best_chunk_size


langsmith_api_key = os.environ.get("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
if not langsmith_api_key:
    LOGGER.warning("Langsmith API key not found. Tracing will be disabled.")


class EmailInfo(BaseModel):
    """Data model for email extracted information."""
    sender: str = Field(..., description="The sender of the email. Store their name and email, if available.")
    sent: str = Field(..., description="The date the email was sent.")
    to: Optional[List[str]] = Field(default_factory=list, description="A list o recipients' names and emails.")
    cc: Optional[List[str]] = Field(default_factory=list, description="A list of additional recipients' names and emails.")
    subject: Optional[str] = Field(None, description="The subject of the email.")
    body: str = Field(..., description="The email body")

local_model = OllamaLLM(model="llama3.1",temperature=0, num_ctx=8192, num_predict=4096, num_gpu=1, stop=["<|eot_id|>"])

parser = JsonOutputParser(pydantic_object=EmailInfo, json_compatible=True)

prompt_template = PromptTemplate.from_template(
"""<|start_header_id|>system<|end_header_id|>
You are an expert assistant who processes email threads with precision and no loss of content.
The input consists of several email strings. Some emails may be duplicates. Your task is:
1. Split the email thread into individual emails using "-***-" as the only delimiter. Each "-***-" marks the end of one complete email.
2. Some email headers may be in different languages, such as Chinese or Russian. Translate the headers to English without changing the date or time and split the email thread. The delimeter "-***-" might not be present in that case
3. Return the unique emails **ONLY** as a JSON list, where each email is a **separate object** with the following fields:

- sender: The sender of the email. Store their name and email, if available, as a string in the format "Name <email@example.com>". If only a name is present, store it as "Name". Extract this after the "From:" field. 
- sent: Extract the full date and time string in the following format: Full weekday name, full month name day, four-digit year, hour:minute:second AM/PM. Be carefull to not change the date or time.
  If the date is not in English, translate it exactly to English without changing the date or time.
- to: A list of recipients' names and emails, if available. Store each entry as a string in the format "Name <email@example.com>" or "Name". The "To" field may contain multiple recipients separated by commas or semicolons but is usually one.
- cc: A list of additional recipients' names and emails. Store each entry as a string in the format "Name <email@example.com>" or "Name". The "Cc" field may contain multiple recipients separated by commas or semicolons.
- subject: The subject of the email, stored as a string. Extract this after the "Subject:" field.
- body: Include the full message text up to but not including the next delimiter "-***-". Do not include "-***-" in the body. Do not summarize or skip any content. If the body is empty, return an empty string.

4. Maintain the chronological order of the emails in the output.
5. Do not hallucinate or add any information that is not present in the email thread. If you are unsure about a date, copy it exactly as shown, translating only the weekday/month names.
6. The length of the JSON list needs to be the same as the number of the emails strictly.
7. Output ONLY the raw JSON array. Do NOT include extra quotes, explanations, or any text.

Process the following email thread:
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
{emails}
<|eot_id|>

<|start_header_id|>assistant<|end_header_id>
"""
)

def load_model(model_id: str = "meta-llama/Llama-3.1-8B-Instruct") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the model and tokenizer for email extraction.

    Args:
        use_accelerate (bool): Whether to use the accelerate library for model loading.
        model_id (str): The model ID to load.

    Returns:
        Tuple[AutoTokenizer, AutoModelForCausalLM]: The loaded tokenizer and model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="sdpa",
        quantization_config=bnb_config,
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True
        ).to("cuda")
    
    model.config.max_length = 8192
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id
        
    
    return model, tokenizer

def load_pipeline(model, tokenizer=None):
    """
    Load the HuggingFace pipeline for text generation.

    Args:
        model (AutoModelForCausalLM) / (Local model): The loaded model.
        tokenizer (AutoTokenizer): The loaded tokenizer.

    Returns:
        pipeline: The HuggingFace pipeline for text generation.
    """

    if tokenizer is None:
        SPLIT_EMAILS_CHAIN = (prompt_template | model | parser)
    else:
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text=False,
            max_new_tokens= 4096,
            do_sample=False,
            temperature=0.0
        )

        llm = HuggingFacePipeline(pipeline=pipe)

        SPLIT_EMAILS_CHAIN = (prompt_template | llm | parser)

    return SPLIT_EMAILS_CHAIN

def split_and_extract_emails_sync(file_path: str) -> List[Dict]:
    """
    Synchronously splits and extracts emails from a .msg file.

    Args:
        file_path (str): Path to the .msg file containing the email thread.
    Returns:
        List[Dict]: A list of dictionaries, each containing the extracted email information.
    """ 

    os.environ["LANGCHAIN_PROJECT"] = "extract_emails_sync"
    
    try:
        #model, tokenizer = load_model()
        #split_emails_chain = load_pipeline(model, tokenizer)
        split_emails_chain = load_pipeline(local_model)
    except Exception as e:
        LOGGER.error(f"Failed to load model or pipeline: {type(e).__name__}: {e}")
        return []

    try:
        raw_msg_content = extract_msg_file(file_path)
        cleaned_msg_content = clean_data(raw_msg_content)
    except Exception as e:
        LOGGER.error(f"Failed to read or clean .msg file '{file_path}': {type(e).__name__}: {e}")
        return []
 
    try:
        LOGGER.info("Splitting emails...")
        splitted_emails = split_email_thread(cleaned_msg_content)[::-1]
    except Exception as e:
        LOGGER.error(f"Failed to split email thread: {type(e).__name__}: {e}")
        return []
    
    raw_model_output = []
    
    ideal_chunk = find_best_chunk_size(len(splitted_emails), 4, 7)

    try:
        for i, chunk in enumerate(chunk_emails(splitted_emails, chunk_size=ideal_chunk)):
            formatted_chunk = "\n*** \n".join(chunk)

            LOGGER.info(f"Processing chunk {i+1}/{(len(splitted_emails) + ideal_chunk - 1) // ideal_chunk}...")
            tic = time()

            outputs = split_emails_chain.invoke(
                {"emails": formatted_chunk},
                config={
                    "metadata": {
                        "chunk_size": len(chunk),
                        "invocation_method": "sync"
                    },
                    "run_name": f"sync_chunk_{i+1}"
                }
            )

            raw_model_output.extend(outputs)
            LOGGER.info(f"Chunk processed in {time() - tic:.2f} seconds.")

        LOGGER.info(f"Total output emails extracted: {len(raw_model_output)}")

    except Exception as e:
        LOGGER.error(f"Failed during model inference: {type(e).__name__}: {e}")
        return []

    return raw_model_output

async def split_and_extract_emails_async(file_path: str) -> List[Dict]:
    """
    Asynchronously splits and extracts emails from a .msg file.

    Args:
        file_path (str): Path to the .msg file.

    Returns:
        List[Dict]: List of extracted email data in chronological order.
    """
    os.environ["LANGCHAIN_PROJECT"] = "extract_emails_async"

    try:
        model, tokenizer = load_model()
        split_emails_chain = load_pipeline(model, tokenizer)
    except Exception as e:
        LOGGER.error(f"Failed to load model or pipeline: {type(e).__name__}: {e}")
        return []

    try:
        raw_msg_content = extract_msg_file(file_path)
        cleaned_msg_content = clean_data(raw_msg_content)
    except Exception as e:
        LOGGER.error(f"Failed to read or clean .msg file '{file_path}': {type(e).__name__}: {e}")
        return []

    try:
        LOGGER.info("Splitting emails...")
        splitted_emails = split_email_thread(cleaned_msg_content)[::-1]
    except Exception as e:
        LOGGER.error(f"Failed to split email thread: {type(e).__name__}: {e}")
        return []

    raw_model_output = []
    ideal_chunk = find_best_chunk_size(len(splitted_emails), 4, 7)

    try:
        chunks = chunk_emails(splitted_emails, chunk_size=ideal_chunk)
        tasks = [process_chunk_async(chunk, ideal_chunk, split_emails_chain) for chunk in chunks]

        LOGGER.info(f"Processing {len(tasks)} chunks asynchronously...")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                LOGGER.error(f"Chunk {i+1} failed: {type(result).__name__}: {result}")
            else:
                raw_model_output.extend(result)

        LOGGER.info(f"Total output emails extracted: {len(raw_model_output)}")

    except Exception as e:
        LOGGER.error(f"Failed during async model inference: {type(e).__name__}: {e}")
        return []

    return raw_model_output


async def process_chunk_async(chunk: List, chunk_size, chain: Runnable) -> List[Dict]:
    chunk = "\n*** \n".join(chunk)
    tic = time()
    result = await chain.ainvoke({"emails": chunk}, config={"metadata": {"chunk_size": len(chunk), "invocation_method": "async"}, "run_name": f"async_chunk_{len(chunk)}"})
    LOGGER.info(f"Chunk {chunk_size} took {time() - tic:.2f}s")
    return result

