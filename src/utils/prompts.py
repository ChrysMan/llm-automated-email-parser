from langchain_core.prompts import PromptTemplate,FewShotPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from typing import Optional, List

class EmailInfo(BaseModel):
    """Data model for email extracted information."""
    sender: str = Field(..., description="The sender of the email. Store their name and email, if available.")
    sent: str = Field(..., description="The date the email was sent.")
    to: Optional[List[str]] = Field(default_factory=list, description="A list o recipients' names and emails.")
    cc: Optional[List[str]] = Field(default_factory=list, description="A list of additional recipients' names and emails.")
    subject: Optional[str] = Field(None, description="The subject of the email.")
    body: str = Field(..., description="The email body")

example_template = """
Input:
{input}

Output:
{output}
"""

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template=example_template
)

headers_cleaning_prompt = """<|start_header_id|>system<|end_header_id|>
You are an email cleaning agent. Your task is to clean email headers while preserving the body.

For the fields "From:", "To:", "Cc:":
1. Each field must contain exactly one identifier per person.
2. If email address or name are present in a field, keep only the email address.
3. If the name is present, keep it as is.
4. If the email address is present, keep it as is.
5. Do not move information between the fields.
6. If the name contains apostrophes ("),(') remove them.

For the field "Sent:":
1. Translate the date to English if it is not in English.
2. Format the date as follows:Full weekday name, full month name day, four-digit year, hour:minute:second AM/PM. Be carefull to not change the date or time.

    
Important Rules:
- Copy body text exactly, no changes. 
- Do NOT hallucinate or add info.  
- Output must start with "From:" and end with sender name + "\n---End of email---".

Example 1:
Input: 
From: John Doe <jdoe@email.com >
Sent: 2023-12-29 22:00
To: "Mary Joe"; "Hara Papadopoulou" <harapap@gmail.com>
Cc: Sales Department 
Subject: Upcoming Shipment
Hello Mary,
This is to inform you about the upcoming shipment.
Thanks and Best regards
John Doe

Output: 
From: jdoe@email.com
Sent: Friday, December 29, 2023 10:00 PM
To: Mary Joe; harapap@gmail.com
Cc: Sales Department
Subject: Upcoming Shipment
Hello Mary,
This is to inform you about the upcoming shipment.
Thanks and Best regards
John Doe
---End of email---

Example 2:
Input: 
Στις Τρίτη, Μαΐου 30, 2023, 11:23 πμ, ο χρήστης Maria Doe <mdoe@email.com> έγραψε:
Καλησπερα Κε Παπαδόπουλε,
Λαβαμε ενημερωση για ένα νέο φορτιο.
Best regards
Maria Doe (Mrs.)

Output: 
From: mdoe@gmail.com
Sent: Tuesday, May 30, 2023 11:23 AM
To: 
Cc:
Subject:
Καλησπερα Κε Παπαδόπουλε,
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

signature_cleaning_prompt = PromptTemplate.from_template("""<|start_header_id|>system<|end_header_id|>
You are an email cleaning agent. You have one task:
                                                         
Remove all the information that follow after the **signature name line**:
1. The signature block starts with phrases like "Best regards", "Thanks and Best regards", "Kind regards", "Sincerely", "Yours faithfully", "Ευχαριστώ", "Ευχαριστώ πολύ", "Με εκτίμηση" or something similar.
2. After this phrase, keep only the senders' name line.
3. Delete everything that appears after that name line, including phone numbers, job titles, company names, addresses, disclaimers, antivirus messages, or blank lines.

Important Rules:
- Copy body text exactly, no changes. 
- Do NOT hallucinate or add info.  
- Output must start with "From:" and end strictly with the greeting + "\n" + senders' name + "\n---End of email---".

Example:
Input: 
From: johndoe@gmail.com
Date: Friday, December 29, 2023 2:09 PM
To: 刘业; Hara Papadopoulou
Cc: Sales Department 
Subject: Upcoming Shipment
Hello Mary,
This is to inform you about the upcoming shipment.
Thanks and Best regards
John
Export Manager
Company XYZ Ltd.
This email has been scanned by XYZ AntiVirus.
***TO AVOID HACKER,PLS RECONFIRM BANK ACCOUNT WITH ME VIA WECHAT OR SKYPE BEFORE ARRANGE PAYMENT***,

Output: 
From: johndoe@gmail.com
Sent: Friday, December 29, 2023 2:09 PM
To: 刘业; Hara Papadopoulou
Cc: Sales Department
Subject: Upcoming Shipment
Hello Mary,
This is to inform you about the upcoming shipment.
Thanks and Best regards
John
---End of email---
                                                         
Process the following email:
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{email}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id>                                                        
"""
)

extraction_prompt = PromptTemplate.from_template(
"""<|start_header_id|>system<|end_header_id|>
You are an expert assistant who processes emails with precision and no loss of content.
The input consists of an email. Your task is:
1. Extract the email thread from the input and return email **ONLY** as a JSON object, with the following fields:

- sender: The sender of the email. Extract this after the "From:" field. 
- sent: Extract the full date and time string. 
- to: A list of recipients' names and emails, if available. 
- cc: A list of additional recipients' names and emails.
- subject: The subject of the email, stored as a string. 
- body: Include the full message text up to the senders' name in the signature. Do not summarize or skip any content. If the body is empty, return an empty string.

2. Some email headers may be in different languages, such as Chinese or Russian. Translate the headers to English without changing the date or time.
3. Do not hallucinate or add any information that is not present in the email thread. If you are unsure about a date, copy it exactly as shown, translating only the weekday/month names.
4. Do NOT include extra quotes, explanations, or any text.

Process the following email thread:
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{email}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id>
"""
)

# Few shot examples
headers_cleaning_examples = [
    {
        "input": """From: John Doe <jdoe@email.com <mailto:jdoe@email.com>>
Sent: Friday, December 29, 2023 2:09 PM
To: "Mary Joe" 
Cc: Sales Department 
Subject: Upcoming Shipment
Hello Mary,
This is to inform you about the upcoming shipment.
Thanks and Best regards
John Doe
Export Manager
Company XYZ Ltd.
This email has been scanned by XYZ AntiVirus.
***TO AVOID HACKER,PLS RECONFIRM BANK ACCOUNT WITH ME VIA WECHAT OR SKYPE BEFORE ARRANGE PAYMENT***""",
        "output": """From: jdoe@email.com
Sent: Friday, December 29, 2023 2:09 PM
To: Mary Joe
Cc: Sales Department
Subject: Upcoming Shipment
Hello Mary,
This is to inform you about the upcoming shipment.
Thanks and Best regards
John Doe
Export Manager
Company XYZ Ltd.
This email has been scanned by XYZ AntiVirus.
***TO AVOID HACKER,PLS RECONFIRM BANK ACCOUNT WITH ME VIA WECHAT OR SKYPE BEFORE ARRANGE PAYMENT***
---End of email---"""
    }
]

"""
embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B", encode_kwargs = {'normalize_embeddings': True})

def create_FewShotPrompt(examples: list, prefix: str):
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        embeddings,
        Chroma,
        k=1,
    )

    return FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=""""""Process the following email:\n{email}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id>"""""",
        input_variables=["email"]
    

"""
