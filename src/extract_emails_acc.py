import json, sys, os
#if "LOCAL_RANK" in os.environ:
#    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch 
import math
import torch.distributed as dist
from time import time
from accelerate import PartialState, Accelerator
from accelerate.utils import gather_object
from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline, BitsAndBytesConfig
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFacePipeline
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Tuple
from utils.logging_config import LOGGER
from utils.graph_utils import extract_msg_file, clean_data, split_email_thread, write_file, append_file, chunk_emails

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
    body: str = Field(..., description="The email body, excluding unnecessary whitespace.")

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

#local_rank = int(os.environ.get("LOCAL_RANK", 0))

#local_rank = 0
#if "LOCAL_RANK" in os.environ:
local_rank = int(os.environ.get("LOCAL_RANK", 0))
#torch.cuda.set_device(0) 

torch.cuda.set_device(local_rank)

model_id = "meta-llama/Llama-3.1-8B-Instruct"

bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16)

state = PartialState()

model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
            #device_map="auto"
            #device_map={"": local_rank}  # place model on local GPU
        )

dist.barrier(device_ids=[local_rank])  # Ensure all processes synchronize before proceeding

#model = model.to(torch.cuda.current_device())
#model = model.to(state.device)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id



pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=local_rank,  # Ensure HF pipeline sends inputs to correct device
    max_new_tokens = 256,
    return_full_text=False,
    do_sample=False,
    temperature=None,
    top_k=None,
    top_p=None
)
#pipe.model.to(state.device)
#pipe.model.to(state.device)

llm = HuggingFacePipeline(pipeline=pipe)

SPLIT_EMAILS_CHAIN = (prompt_template | llm | parser)

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

    #state.wait_for_everyone()
    #wait_for_all(state)

    raw_msg_content = extract_msg_file(file_path)
    cleaned_msg_content = clean_data(raw_msg_content)
    LOGGER.info("Splitting emails...")
    try:
        splitted_emails = split_email_thread(cleaned_msg_content)[::-1]

        chunk_size= math.ceil(len(splitted_emails) / 8)
        chunks = list(chunk_emails(splitted_emails, chunk_size=chunk_size))
        joined_chunks = ["\n***\n".join(chunk) for chunk in chunks]

        
        with state.split_between_processes(joined_chunks) as local_chunks:
            results = []

            for idx, chunk in enumerate(local_chunks):
                result = SPLIT_EMAILS_CHAIN.invoke({"emails": chunk}, config={"metadata": {"chunk_num": idx, "invocation_method": "distributed inference"}, "run_name": f"chunk_{idx}"})
                results.extend(result)

        #accelerator.wait_for_everyone()
        #wait_for_all(accelerator)
        state.wait_for_everyone()

        all_results = gather_object(results)
        
        if state.is_main_process:
            flat_results = [item for sublist in all_results for item in sublist]
            LOGGER.info(f"Total output emails: {len(flat_results)}")
            return flat_results
        else:
            return []
    
    except Exception as e:
        LOGGER.error(f"Failed to split emails: {e}")

    return []


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if len(sys.argv) != 2:
        LOGGER.error("Usage: python emailParsing.py <dir_path>")
        sys.exit(1)
    
    dir_path = sys.argv[1]

    if not os.path.isdir(dir_path):
        LOGGER.error(f"{dir_path} is not a valid directory.")
        sys.exit(1)

    folder_name = os.path.basename(os.path.normpath(dir_path))

    output_path = os.path.join(dir_path, f"{folder_name}.json")

    email_data = []
    #graph = nx.DiGraph()
    for filename in os.listdir(dir_path):
        if filename.endswith(".msg"):
            file_path = os.path.join(dir_path, filename)
            try:

                tic = time()
                data = split_and_extract_emails_acc(file_path)
                LOGGER.info(f"Time taken to process {filename}: {time() - tic} seconds")

                #graph = build_email_graph(graph, data, filename)
                email_data.extend(data) 
            except Exception as e:
                LOGGER.error(f"Processing {filename} failed: {e}")
    
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(email_data, file, indent=4, ensure_ascii=False, default=str)
    