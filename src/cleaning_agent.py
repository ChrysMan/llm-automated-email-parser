import json, os, sys
import ray
import traceback
from time import time
from ray.util.actor_pool import ActorPool
from langsmith import trace, Client, traceable
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_core.prompts import PromptTemplate
from typing import Optional, List, Dict, Tuple
from pydantic import BaseModel, Field
from utils.logging_config import LOGGER
from utils.graph_utils import write_file, append_file

langsmith_api_key = os.environ.get("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "email_cleaning"
if not langsmith_api_key:
    LOGGER.warning("Langsmith API key not found. Tracing will be disabled.")

prompt = PromptTemplate.from_template(
"""<|start_header_id|>system<|end_header_id|>
You are an email cleaning agent. You have two tasks:
1. For the fields "From:", "To:", "Cc:":
    - If both a name and an email are present, keep only the email.
    - If only one of them is present, leave it as is. 
    - If the name contains apostrophes ("'") remove them. 
    - Output all addresses in a consistent format, separated by semicolons.

2. Remove footer or disclaimer sections that follow after the **signature name line**. 
    - The signature block starts with phrases like "Best regards", "Thanks and Best regards", "Kind regards", "Sincerely", "Yours faithfully", "Ευχαριστώ", "Με εκτίμηση" etc.
    - After the signature keep only the name (e.g., "John Doe"), remove all other text like job titles, company addresses, legal disclaimers, antivirus checks, etc.

---Important Rules---
* Preserve the rest of the email exactly as it is.
* Only clean the header fields and remove unnecessary disclaimers or auto-added footers after the signature name.
* Output only the email text without any additional formatting or explanations. 
* The output should start with "From:" and end with the name of the sender *strictly*.

---Example---
Input:
    From: John Doe <jdoe@email.com <mailto:jdoe@email.com>>
    To: Mary Joe <mjoe@email.com>
    Cc: Sales Department <sales@company.com>
    Subject: Upcoming Shipment
    Hello Mary,
    This is to inform you about the upcoming shipment.
    Best regards
    John Doe
    Export Manager
    Company XYZ Ltd.
    This email has been scanned by XYZ AntiVirus.

Output:
    From: jdoe@email.com
    To: mjoe@email.com
    Cc: sales@company.com
    Subject: Upcoming Shipment
    Hello Mary,
    This is to inform you about the upcoming shipment.
    Best regards
    John Doe

Process the following email:
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
{email}
<|eot_id|>

<|start_header_id|>assistant<|end_header_id>

"""
)

model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="float16",
    device_map="cuda:0"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

@traceable(name="SE-230054-7")
def clean_email(email_text:str) -> str:
    """Cleans the email text by removing unnecessary information and formatting."""
    try:
        # Prepare the prompt
        prompt_text = prompt.format(email=email_text)

        # Tokenize
        input = tokenizer(prompt_text, return_tensors="pt")
        input = input.to("cuda:0")  # Move to the correct device

        token_ids = tokenizer.encode(email_text)
        token_count = len(token_ids)
        max_new_tokens = next_power_of_two(token_count)
        print(f"Token count: {token_count}, Max new tokens: {max_new_tokens}\n\n")
        print("<|eot_id|>: ", tokenizer.encode("<|eot_id|>", add_special_tokens=False))
        
        # Generate the cleaned email text
        cleaned_email = model.generate(
            input_ids=input['input_ids'],
            attention_mask=input['attention_mask'],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id = tokenizer.eos_token_id
        )
        
        # Decode the generated text
        cleaned_email_text = tokenizer.decode(cleaned_email[0], skip_special_tokens=False)
        #print("Raw response:\n", cleaned_email_text)
        real_response = cleaned_email_text.split("<|start_header_id|>assistant<|end_header_id>")[-1].split("<|eot_id|>")[0].replace("<|start_header_id|><|endoftext|>", "").strip()
       
        return real_response
    except Exception as e:
        LOGGER.error(f"Error cleaning email: {e}")
        return email_text  # Return original if error occurs
    
def next_power_of_two(x: int) -> int:
    if x <= 0:
        return 1
    return 1 << (x - 1).bit_length()

if __name__ == "__main__":
    email_text = """Your email text goes here"""

    cleaned = clean_email(email_text)
    print(cleaned)