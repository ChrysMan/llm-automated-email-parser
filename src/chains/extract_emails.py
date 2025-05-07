import asyncio, os
from typing import List
from utils.logging_config import LOGGER
from utils.graph_utils import extract_msg_file, clean_data, split_email_thread, write_file, append_file, chunk_emails
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from time import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from langsmith import Client

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

model = OllamaLLM(model="llama3.1",temperature=0, num_ctx=32768, num_predict=16384)
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
6. Clean any unnecessary whitespace and new lines.
7. Output only the JSON list of emails without any additional text or explanation.

Process the following email thread:
<|eot_id|><|start_header_id|>user<|end_header_id|>
{emails}
<|eot_id|><|start_header_id|>assistant<|end_header_id>
"""
)

SPLIT_EMAILS_CHAIN = (prompt_template | model | parser)

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

    raw_msg_content = extract_msg_file(file_path)
    cleaned_msg_content = clean_data(raw_msg_content)
    write_file(cleaned_msg_content, "clean1.txt")
    LOGGER.info("Splitting emails...")
    try:
        raw_model_output = []
        splitted_emails = split_email_thread(cleaned_msg_content)
        reversed_list = splitted_emails[::-1]
        chunk_size = 7
        for chunk in chunk_emails(reversed_list, chunk_size=chunk_size):
            chunk = "\n*** \n".join(chunk)
            append_file(chunk, "emailSplit.txt")

            # Invoke the model with the chunk
            tic = time()
            raw_model_output.extend(SPLIT_EMAILS_CHAIN.invoke({"emails": chunk}, config={"metadata": {"chunk_size": chunk_size, "invocation_method": "sync"}, "run_name": f"sync_chunk_{chunk_size}"}))
            LOGGER.info(f"Time taken for chunk: {time() - tic:.2f} seconds")
            
        LOGGER.info("Splitted emails...")        
    
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

    raw_msg_content = extract_msg_file(file_path)
    cleaned_msg_content = clean_data(raw_msg_content)
    write_file(cleaned_msg_content, "clean1.txt")
    LOGGER.info("Splitting emails...")
    try:
        raw_model_output = []
        chunk_size=7
        splitted_emails = split_email_thread(cleaned_msg_content)
        reversed_list = splitted_emails[::-1]
        chunks = chunk_emails(reversed_list, chunk_size=chunk_size)
        tasks = [process_chunk(chunk, chunk_size) for chunk in chunks]
        results = await asyncio.gather(*tasks)
        for result in results:
            raw_model_output.extend(result)           
        
        LOGGER.info("Splitted emails...")        
    
    except Exception as e:
        LOGGER.error(f"Failed to split emails: {e}")

    return raw_model_output

async def process_chunk(chunk: List, chunk_size) -> List[Dict]:
    chunk = "\n*** \n".join(chunk)
    tic = time()
    result = await SPLIT_EMAILS_CHAIN.ainvoke({"emails": chunk}, config={"metadata": {"chunk_size": chunk_size, "invocation_method": "async"}, "run_name": f"async_chunk_{chunk_size}"})
    LOGGER.info(f"Time taken for chunk: {time() - tic:.2f} seconds")
    return result