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
from utils.graph_utils import write_file, append_file, clean_data

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
    - Whatever format you choose for one field must be strictly applied to the rest of the fields, separated by semicolons.
    - If any field contains only a name, keep only the name.
    - If any field contains an email address, keep only the email address.
    - If they contain both a name and an email address, keep only the email address.
    - If the name contains apostrophes ("'") remove them.

2. Remove all the information that follow after the **signature name line**. 
    - The signature block starts with phrases like "Best regards", "Thanks and Best regards", "Kind regards", "Sincerely", "Yours faithfully", "Ευχαριστώ", "Ευχαριστώ πολύ", "Με εκτίμηση" or something similar.
    - After this phrase, keep only the very next line if it contains the sender’s name.
    - Delete everything that appears after that name line — including phone numbers, job titles, company names, addresses, disclaimers, antivirus messages, or blank lines.

---Important Rules---
* *DO NOT* hallucinate or add any information that is not present in the email.
* Output only the email text without any additional formatting or explanations. 
* The output should start with "From:" and end with the name of the sender followed by the phrase "---End of email---" once *strictly*.

---Example 1---
Input:
    From: John Doe <jdoe@email.com <mailto:jdoe@email.com>>
    To: Mary Joe 
    Cc: Sales Department 
    Subject: Upcoming Shipment
    Hello Mary,
    This is to inform you about the upcoming shipment.
    Best regards
    John Doe
    Export Manager
    Company XYZ Ltd.
    This email has been scanned by XYZ AntiVirus.

Output:
    From: John Doe
    To: Mary Joe
    Cc: Sales Department
    Subject: Upcoming Shipment
    Hello Mary,
    This is to inform you about the upcoming shipment.
    Best regards
    John Doe 
    ---End of email---

---Example 2---
Input:
    Στις Τρίτη, Μαΐου 30, 2023, 11:23 πμ, ο χρήστης Maria Doe <mdoe@email.com> έγραψε:
    Καλησπερα Κε Πετρόπουλε,
    Πως ειστε ?
    Λαβαμε ενημερωση για ένα νέο φορτιο.
    Best regards
    Maria Doe (Mrs.)
    Operation manager
    Company Name S.A.
    23B Oxford street | London-England 64336

Output:
    Στις Τρίτη, Μαΐου 30, 2023, 11:23 πμ, ο χρήστης mdoe@email.com έγραψε:
    Καλησπερα Κε Πετρόπουλε,
    Πως ειστε ?
    Λαβαμε ενημερωση για ένα νέο φορτιο.
    Best regards
    Maria Doe (Mrs.)
    ---End of email---

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
    device_map="cuda:2"
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
        input = input.to("cuda:2")  # Move to the correct device

        token_ids = tokenizer.encode(email_text)
        token_count = len(token_ids)
        max_new_tokens = next_power_of_two(token_count)
        print(f"Token count: {token_count}, Max new tokens: {max_new_tokens}\n\n")
        print("<|eot_id|>: ", tokenizer.encode("<|eot_id|>", add_special_tokens=False))
        print("\n\n")
        
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
        real_response = cleaned_email_text.split("<|start_header_id|>assistant<|end_header_id>")[-1].split("---End of email---")[0].strip()
        cleaned_response = clean_data(real_response)
        return cleaned_response
    except Exception as e:
        LOGGER.error(f"Error cleaning email: {e}")
        return email_text  # Return original if error occurs
    
def next_power_of_two(x: int) -> int:
    if x <= 0:
        return 1
    return 1 << (x - 1).bit_length()

if __name__ == "__main__":
    email_text = """Στις Τρίτη, Μαΐου 30, 2023, 11:23 πμ, ο χρήστης Marina Koletzaki έγραψε:
Καλησπερα Κε Κουτσουβαλα,
Πως ειστε ?
Λαβαμε ενημερωση για ένα νέο φορτιο αεροπορικο ως παρακατω.
Πειτε μου εάν το προχωραμε .
We got a booking for M.IOANNO, below is the details, pls let me know if we can go ahead, thanks.
S/DALIAN GOLDEN-CAT
C/M.IOANNO
FOB Term
22ctns/212kgs/0.6cbm
Ready date is on 30TH MAY
From DLC to ATH
CA: via BJS, from BJS to ATH is directly service, D3,7/WEEK
+100KGS: USD3.62/kg all-in (spot rate for dense cargo 1cbm>300kgs)
QR: via BJS and DOH
+100KGS: USD3.97/kg all-in
The cost cannot be locked, any change will send to you.
Επισης και τα τοπικα Ελλαδος παρακατω : 
*	Δικ.διατακτικης : 35 ευρω
*	Πρακτορειακα : 45 ευρω
Εν αναμονη απαντησης σας.
Best regards
Marina Koletzaki (Mrs.)
Operation manager
Arian Maritime S.A.
133Α Filonos street | Piraeus-Greece 18536
"""

    cleaned = clean_email(email_text)
    print(cleaned)