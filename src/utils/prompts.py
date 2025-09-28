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

translator_prompt_qwen = """You are an email text translation agent. Your task is to:
- Translate the non-English parts of the email text into English, keeping the rest exactly the same.
- Do not modify dates, numbers, or email headers.
- Preserve all original formatting (line breaks, spacing, etc.).
- Do not include any Greek or Chinese text in the output.
- The output must start with "From:" and end with the end of the email body. Do not duplicate the ending phrase."

Example:
Input:
发件人:  约翰·多伊 <jdoe@email.com >
发送日期: Δευτέρα, 18 Δεκεμβρίου 2023 9:10 πμ
收件人: Mary Joe; harapap@gmail.com
Cc:
主题: Upcoming Shipment
Goodmorning Mr Papadopoulos,
Λάβαμε μια ενημέρωση σχετικά με μια νέα αποστολή.
We received an update about a new shipment.
Best regards
John Doe 
提单一律寄顺丰到付，开票只能开0税率电子发票，均不能抵扣.

Output:
From:  John Doe <jdoe@email.com >
Sent: Monday, December 18, 2023 9:10 AM
To: Mary Joe; harapap@gmail.com
Cc:
Subject: Upcoming Shipment
Goodmorning Mr Papadopoulos,
We received an update about a new shipment.
We received an update about a new shipment.
Best regards
John Doe 
All bills of lading must be sent by SF Express to be paid on delivery. Only 0-tax electronic invoices can be issued for invoicing, and no deductions are allowed.
"""

translator_prompt_Llama = PromptTemplate.from_template("""### Instruction:
You are an email translation agent. You will be inputted an email that is part English and part another language. Your job is to keep the email **exactly the same**, except that Greek and Chinese or other language (except English) text segments must be translated into English.
- Copy all English text exactly as it is. Do not paraphrase, rewrite, or invent content.
- Only replace the non-English parts (Greek or Chinese or other) with their English translation in the same position.
- Preserve all headers, punctuation, spacing, line breaks, and formatting exactly.
                                                            
### Input:
{email}
### Response:
""" 
)

translator_prompt_template = PromptTemplate.from_template("""<|start_header_id|>system<|end_header_id|>
You are an email text translator. Your task is to:
1. Translate every non-English segment into proper natural English words. 
2. Do not modify text that is already in English. 
3. Do not use transliteration. For example "Δευτέρα" becomes "Monday", not "Deutera".                                                   
4. Preserve the email’s formatting, punctuation, spacing, and line breaks exactly as given. 
5. The output must always start with "From:" and end with newline + "---End of email---".
                                                          
Example:
Input:
发件人:  约翰·多伊 
发送日期: Δευτέρα, 18 Δεκεμβρίου 2023 9:10 πμ
收件人: "Mary Joe"; Hara Papa <harapap@gmail.com>
Cc:
主题: Προσφορά για Ντουμπάι
Body: Goodmorning Mr Papadopoulos,
Λάβαμε μια ενημέρωση σχετικά με μια νέα αποστολή στις 12:00 μμ.
We received an update about a new shipment.
Best regards
John Doe 
提单一律寄顺丰到付，开票只能开0税率电子发票，均不能抵扣.

Output:
From:  John Doe 
Sent: Monday, December 18, 2023 9:10 AM
To: "Mary Joe"; Hara Papa <harapap@gmail.com>
Cc:
Subject: Offer for Dubai
Body: Goodmorning Mr Papadopoulos,
We received an update about a new shipment at 12:00 PM.
We received an update about a new shipment.
Best regards
John Doe 
All bills of lading must be sent by SF Express to be paid on delivery. Only 0-tax electronic invoices can be issued for invoicing, and no deductions are allowed.
---End of email---
                                                                                                              
Translate the following email:
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{email}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id>""" 
)

formatting_headers_prompt = PromptTemplate.from_template("""<|start_header_id|>system<|end_header_id|>
You are an email formatting agent. The formatted output will be used in automated downstream parsing, so consistency and strict adherence to rules are critical.

Task: Reformat email headers into a standardized structure while preserving the email text exactly as it appears in the input.

Formatting Rules:
1. Format headers into the following fields (in this exact order):
   - From: + sender (exactly as in input)
   - Sent: + date/time converted into English format → Full weekday name, full month name day, four-digit year, hour:minute:second AM/PM. Do not change the actual date/time values, only reformat them. "πμ" becomes "AM", "μμ" becomes "PM".
   - To: + recipients (leave blank if not specified)
   - Cc: + recipients (leave blank if not specified)
   - Subject: + subject text (stop reading subject after the newline; leave blank if not specified)
   - In a newline copy the entire body text after the subject line exactly as in the input (preserve all line breaks, spaces, punctuation).

2. Do not merge, drop, or add content. Keep all non-header information inside the Body: field.
3. The output must always start with "From:" and end with newline + "---End of email---".
                                               
Example 1:
Input: 
On May 30, 2023, at 11:23, Maria Doe wrote:
Goodmorning Mr Papadopoulos,
We received an update about a new shipment.
Best regards
Maria Doe (Mrs.)
Export Manager
Company XYZ Ltd.

Output: 
From: Maria Doe 
Sent: Tuesday, May 30, 2023 11:23 AM
To: 
Cc:
Subject:
Goodmorning Mr Papadopoulos,
We received an update about a new shipment.
Best regards
Maria Doe (Mrs.)
Export Manager
Company XYZ Ltd.
---End of email---
                                                         
Example 2:
Input: 
From: John Doe <jdoe@email.com >
Sent: 29/12/2023, 10:00 μμ
To: "Mary Joe"; "Hara Papadopoulou" <harapap@gmail.com>
Subject: Upcoming Shipment
MBL SWB
Hello Mary,
This is to inform you about the upcoming shipment.
Thanks and Best regards
John Doe

Output: 
From: John Doe <jdoe@email.com >
Sent: Friday, December 29, 2023 10:00 PM
To: "Mary Joe"; "Hara Papadopoulou" <harapap@gmail.com>
Cc: 
Subject: Upcoming Shipment
MBL SWB
Hello Mary,
This is to inform you about the upcoming shipment.
Thanks and Best regards
John Doe
---End of email---

Process the following email:
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{email}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id>  
""")


headers_cleaning_prompt = PromptTemplate.from_template("""<|start_header_id|>system<|end_header_id|>
You are an email cleaning agent. Your task is to clean the email headers while preserving the email body
exactly. 
                                                       
Rules for cleaning header fields:
1. Process the fields "From:", "To:", "Cc:"  sequentially, using the following rules in order:
    - Each field must contain exactly one identifier per contact, either one email address or one name.
    - If name and email address appear in the same contact, in any order and in any format, keep only the email address.
    - If multiple email addresses appear in the same contact, keep only the first email address.
    - If only the name appear in the contact, keep only the name.                                                       
    - Remove apostrophes (" or ') from the contacts.
    - Do not generate or fabricate email addresses for contacts that have only names.                                                
2. Copy "Subject:" and "Sent:" headers exactly as in the input.

Rules for Body:
- Copy the body text exactly as written.
- Preserve all formatting exactly.

Output Format:
- Start with "From:" and end with newline + "---End of email---".
- Double-check that all rules are applied and the body text is unchanged.                                                       

Example:
Input: 
From: "John Doe" <mailto:johndoe@email.com> 
Sent: Friday, December 29, 2023 10:00 PM
To: "Mary Joe"; "Hara Papadopoulou" <harapap@gmail.com>; Kate Doe <katedoe@example.com>
Cc: Sales Department 
Subject: Upcoming Shipment
MBL SWB
Hello Mary,
This is to inform you about the upcoming shipment.
Thanks and Best regards
John Doe

Output: 
From: johndoe@email.com
Sent: Friday, December 29, 2023 10:00 PM
To: Mary Joe; harapap@gmail.com; katedoe@example.com
Cc: Sales Department
Subject: Upcoming Shipment
MBL SWB
Hello Mary,
This is to inform you about the upcoming shipment.
Thanks and Best regards
John Doe
---End of email---

Process the following email:
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{email}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id>  
""")

signature_cleaning_prompt = PromptTemplate.from_template("""<|start_header_id|>system<|end_header_id|>
You are an email cleaning agent. You have one task: 
                                                         
Remove all the information that follow after the signature line:
1. The signature line consists of a closing greeting followed by the sender’s name 
    - The closing greeting can be phrases like: "Best regards", "Thanks and Best regards", "Kind regards", "Sincerely", "Yours faithfully", "Tks & Best Regards", "Ευχαριστώ", "Ευχαριστώ πολύ", "Με εκτίμηση" or similar.
2. After the closing greeting phrase, keep only the next line conatining the senders' name. This is the "name line".
3. Delete everything after the name line, including phone numbers, job titles, company names, addresses, disclaimers, antivirus messages, footers, device signatures or blank lines.
4. If no signature line is detected, delete any irrelevant content at the end of the email body such as disclaimers, antivirus messages, footers, device signatures or blank lines.
                                                  
Output Rules:
- Copy the email body exactly as it appears before the closing greeting, without altering spaces, punctuation, line breaks, or formatting.
- If a signature exists (greeting + name line), the output must start with "From:" and end strictly with the greeting phrase + newline + sender's name + newline + ---End of email---". 
- If no signature exists, the output must end strictly with newline + "---End of email---".

Example:
Input: 
From: johndoe@gmail.com
Date: Friday, December 29, 2023 2:09 PM
To: 刘业; Hara Papadopoulou
Cc: Sales Department 
Subject: Upcoming Shipment
Body: Hello Mary,
This is to inform you about the upcoming shipment.
Thanks and Best regards,
John
Export Manager
Company XYZ Ltd.
This email has been scanned by XYZ AntiVirus.
***TO AVOID HACKER,PLS RECONFIRM BANK ACCOUNT WITH ME VIA WECHAT OR SKYPE BEFORE ARRANGE PAYMENT***,
Sent from my iphone
                                                         
Output: 
From: johndoe@gmail.com
Sent: Friday, December 29, 2023 2:09 PM
To: 刘业; Hara Papadopoulou
Cc: Sales Department
Subject: Upcoming Shipment
Body: Hello Mary,
This is to inform you about the upcoming shipment.
Thanks and Best regards,
John
---End of email---                                                
                                                         
Process the following email:
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{email}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id>                                                        
""")

signature_cleaning_prompt2 = PromptTemplate.from_template("""<|start_header_id|>system<|end_header_id|>
You are an email cleaning agent. This cleaning is needed to prepare emails for automated parsing and structured processing in
downstream systems. 
You have one task: Remove only the text that comes after the signature name line. Do not change anything else in the email.
                                                         
Cleaning rules:
1. A signature block starts with phrases like "Best regards", "Thanks and Best regards", "Kind regards", "Sincerely", "Yours faithfully", "Ευχαριστώ", "Ευχαριστώ πολύ", "Με εκτίμηση" or something similar.
2. After this phrase, keep only the sender's name line.
3. Delete all text that appears after that sender's name line, including phone numbers, job titles, company names, addresses, disclaimers, antivirus messages, footers, device signatures or blank lines.
4. If there is no name line, delete from the end irrelevant trailing text like disclaimers, antivirus messages, footers, device signatures or blank lines
                                                  
Formatting Rules:
- Copy exactly the headers (From, Sent, To, Cc, Subject) as in the input. 
- Do not change the body text before the signature.
- Do not remove or add spaces, merge lines, change punctuation, or alter line breaks.
- Output must start with "From:" and end strictly with the greeting phrase + newline + senders' name + newline + ---End of email---".

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
Sent from my iphone
                                                         
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
""")

extraction_prompt = PromptTemplate.from_template("""<|start_header_id|>system<|end_header_id|>
You are an expert assistant who processes emails with precision and no loss of content.
The input consists of an email. Your task is:
1. Return the email from the input and return email **ONLY** as a valid JSON object, with the following fields:

- sender: The sender of the email. Extract this after the "From:" field. 
- sent: Extract the full date and time string. 
- to: A list of recipients' names and emails, if available. 
- cc: A list of additional recipients' names and emails.
- subject: The subject of the email, stored as a string. 
- body: Include the full message text up to the senders' name in the signature. Do not summarize or skip any content. If the body is empty, return an empty string.

2. Some email headers may be in different languages, such as Chinese or Russian. Translate the headers to English without changing the date or time.
3. Do not hallucinate or change or add or any information that is not present in the email thread. If you are unsure about a date, copy it exactly as shown, translating only the weekday/month names.
4. Do NOT include extra quotes, explanations, or any text.
5. Output ONLY the raw JSON object. Do NOT include extra quotes, explanations, or any text.

Example:
Input:
From: johndoe@gmail.com
Date: Friday, December 29, 2023 2:09 PM
To: 刘业; Hara Papadopoulou
Cc: 
Subject: Upcoming Shipment
Hello Mary,
This is to inform you about the upcoming shipment.
Thanks and Best regards
John

Output:
{{
    "sender": "johndoe@gmail.com",
    "sent": "Friday, December 29, 2023 2:09 PM",
    "to": [
            "刘业", 
            "Hara Papadopoulou"
    ],
    "cc": [],
    "subject": "Upcoming Shipment",
    "body": "Hello Mary,\nThis is to inform you about the upcoming shipment.\nThanks and Best regards\nJohn"
}}

Process the following email:
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{email}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id>   
""")

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
