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
3. The output must always start with "From:" and end with newline + "End of email".
                                               
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
End of email
                                                         
Example 2:
Input: 
From: John Doe <jdoe@email.com >
Sent: 29/12/2023, 22:00 μμ
To: "Mary Joe"; "Hara Papadopoulou" <harapap@gmail.com>
Subject: Upcoming Shipment
MBL SWB
Hello Mary,
This is to inform you about the upcoming shipment.
Thanks and Best regards
John Doe

Output: 
From: John Doe <jdoe@email.com >
Sent: Friday, December 29, 2023 22:00 PM
To: "Mary Joe"; "Hara Papadopoulou" <harapap@gmail.com>
Cc: 
Subject: Upcoming Shipment
MBL SWB
Hello Mary,
This is to inform you about the upcoming shipment.
Thanks and Best regards
John Doe
End of email

Process the following email:
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{email}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id>  
"""
)

translator_prompt_template = PromptTemplate.from_template("""<|start_header_id|>system<|end_header_id|>
You are an email text translator. Your task is to:
1. Translate every non-English segment into proper natural English words. 
2. Do not modify text that is already in English. 
3. Do not use transliteration. For example "Δευτέρα" becomes "Monday", not "Deutera".                                                   
4. Preserve the email’s formatting, punctuation, spacing, and line breaks exactly as given. 
5. The output must always start with "From:" and end strictly with a newline + "End of email".
                                                          
Example:
Input:
发件人:  约翰·多伊 
发送日期: Δευτέρα, 18 Δεκεμβρίου 2023 9:10 πμ
收件人: "Mary Joe"; Hara Papa <harapap@gmail.com>
Cc:
主题: Προσφορά για Ντουμπάι
Goodmorning Mr Papadopoulos,
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
Goodmorning Mr Papadopoulos,
We received an update about a new shipment at 12:00 PM.
We received an update about a new shipment.
Best regards
John Doe 
All bills of lading must be sent by SF Express to be paid on delivery. Only 0-tax electronic invoices can be issued for invoicing, and no deductions are allowed.
End of email
                                                                                                              
Translate the following email:
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{email}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id>""" 
)

# You are an email cleaning agent. Your task is to clean the email body by removing signatures and irrelevant endings while preserving the email headers and body text exactly as in the input.
                                                         
# Rules:
# 1. Remove everything that appears after the name line (sender's name), including phone numbers, job titles, company names, addresses, disclaimers, antivirus messages, footers, device signatures or blank lines.
# 4. If there is no name line delete from the end, text that seems irrelevant to the email body like disclaimers, antivirus messages, footers, device signatures or blank lines
                                                  
# Important Rules:
# - Copy the body text exactly as it is written, trimming only according to the above rules. Do not remove or add spaces, do not merge lines, do not change punctuation, and do not alter line breaks. Preserve all formatting exactly. 
# - Output must start with "From:" and end strictly with the greeting + newline + name line + newline + "---End of email---".
# - If there is no greeting + name line, output must start with "From:" and end strictly with the last valid body line + newline + "---End of email---".

# Rules for identifying the signature block and removing content:                                                       
# 1. The signature block begins with a closing greeting phrase and is followed by the sender's name. 
#     - The closing greeting phrase can be phrases like: "Best regards", "Thanks and Best regards", "Kind regards", "Sincerely", "Yours faithfully", "Tks & Best Regards", "Ευχαριστώ", "Ευχαριστώ πολύ", "Με εκτίμηση" or similar.
# 2. After the closing greeting phrase, keep only the line after it that contains the sender's name. This is the "name line".
# 3. Remove everything after the name line, including phone numbers, job titles, company names, addresses, disclaimers, antivirus messages, footers, device signatures or blank lines.
# 4. If no signature block is detected, remove from the end any irrelevant trailing text like disclaimers, antivirus messages, footers, device signatures or blank lines.
                                                  
# Output Rules:
# - Copy exactly the headers (From, Sent, To, Cc, Subject) as in the input. 
# - Copy the body text exactly as in the input, trimming only according to the above rules. Do NOT invent or add new sentences to the email body.
# - If no signature block exists, the output must start with "From" and end with newline + "---End of email---".
# - If a signature block exists (greeting phrase + name line), the output must start with "From:" and end with the greeting phrase + newline + sender's name + newline + ---End of email---". 

# You are an email cleaning agent. 
# Your only task: remove everything that comes after the greeting block in the email body. 
# If no greeting block exists, keep the body exactly as written and only remove irrelevant endings (e.g. disclaimers, antivirus notices, device signatures, or blank trailing lines).

# Greeting block rules:
# 1. The greeting block begins with a closing greeting phrase (e.g., "Best regards", "Thanks and Best regards", "Kind regards", "Sincerely", "Yours faithfully", "Tks & Best Regards", "Ευχαριστώ", "Ευχαριστώ πολύ", "Με εκτίμηση" or similar).
# 2. It ends with the sender's name that comes in the line after the closing greeting phrase.
                                                                                                                                                                
# Output Rules:
# 1. Delete everything that appears after the greeting block, such as phone numbers, job titles, company names, addresses, disclaimers, antivirus messages, footers, device signatures or blank lines.
# 2. If a greeting block exists → output must start with "From:" and end strictly with:
# closing greeting phrase + newline + sender's name + newline + "---End of email---"
# 3. If no greeting block exists → output must start with "From:" and end strictly with:
# last valid body line + newline + "---End of email---"

cleaning_prompt = PromptTemplate.from_template("""<|start_header_id|>system<|end_header_id|>
You are an email cleaning agent. You have two tasks:
1. Remove irrelevant text from the end of the email body      
2. Remove only apostrophes (" or ') from the contacts in the headers.     

Rules for task 1:
(1) A signature block begins with a closing greeting "Best regards", "Thanks and Best regards", "Kind regards", "Sincerely", "Yours faithfully", "Tks & Best Regards", "Ευχαριστώ", "Ευχαριστώ πολύ", "Με εκτίμηση" or similar) and ends with the sender's name on the next line.  
(2) Delete everything after the sender's name: phone numbers, job titles, company names, disclaimers, antivirus notices, footers, or device signatures.  
(3) If no closing greeting + name is found, remove only irrelevant trailing text (like disclaimers, antivirus notices, footers, device signatures). 
(4) If you find irrelevant text like notes, disclaimers etc before the signature block, remove it with caution, only if it is clearly not part of the email body.
(5) Do not invent, rewrite, or add content. Keep all spacing, line breaks, punctuation, and formatting exactly as in the input.  

Rules for task 2:                             
(1) Process the fields "From:", "To:", "Cc:"  sequentially, using the following rules in order:
    - Remove the duplicate emails and names from the contacts.                                                     
    - Remove apostrophes (" or ') from the contacts.                                            
(2) Copy "Subject:" and "Sent:" headers exactly as in the input.
                                               
Output format:  
- Preserve the body until the cut point.  
- End the output with:  
   - If a signature block exists → sender's name + newline + "End of email"  
   - Otherwise -> newline + "End of email" 
                                               
Example:
Input: 
From: johndoe@email.gr
Sent: Friday, December 29, 2023 22:00 PM
To: Mary Joe/Ms. Mary Joe; harapap@gmail.com <harapap@gmail.com>; Kate Doe <katedoe@example.com> 
Cc: Sales Department
Subject: Upcoming Shipment
Hello Mary,
This is to inform you about the upcoming shipment.
Thanks and Best regards,
John Doe
Export Manager
Company XYZ Ltd.
190, Venizelos Str 12345 Salamina – Greece
This email has been scanned by XYZ AntiVirus.
***TO AVOID HACKER,PLS RECONFIRM BANK ACCOUNT WITH ME VIA WECHAT OR SKYPE BEFORE ARRANGE PAYMENT***,
Sent from my iphone
                                                         
Output: 
From: johndoe@email.gr
Sent: Friday, December 29, 2023 22:00 PM
To: Mary Joe; harapap@gmail.com <harapap@gmail.com>; Kate Doe <katedoe@example.com> 
Cc: Sales Department
Subject: Upcoming Shipment
Hello Mary,
This is to inform you about the upcoming shipment.
Thanks and Best regards,
John Doe
End of email                                              
                                                         
Process the following email:
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{email}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id>                                                      
"""
)                 

signature_cleaning_prompt = PromptTemplate.from_template("""<|start_header_id|>system<|end_header_id|>
You are an email cleaning agent. Your job is to remove irrelevant text from the end of the email body while keeping the original headers and message intact.  

Rules:  
1. A signature block begins with a closing greeting "Best regards", "Thanks and Best regards", "Kind regards", "Sincerely", "Yours faithfully", "Tks & Best Regards", "Ευχαριστώ", "Ευχαριστώ πολύ", "Με εκτίμηση" or similar) and ends with the sender's name on the next line.  
2. Delete everything after the sender's name: phone numbers, job titles, company names, disclaimers, antivirus notices, footers, or device signatures.  
3. If no closing greeting + name is found, remove only irrelevant trailing text (like disclaimers, antivirus notices, footers, device signatures).  
4. Do not invent, rewrite, or add content. Keep all spacing, line breaks, punctuation, and formatting exactly as in the input.  

Output format:  
- Copy the headers (From, Sent, To, Cc, Subject) exactly as in the input..  
- Preserve the body until the cut point.  
- End the output with:  
   - If a signature block exists → sender's name + newline + "End of email"  
   - Otherwise → newline + "End of email" 
                                                         
Example:
Input: 
From: johndoe@gmail.com
Sent: Friday, December 29, 2023 14:09 PM
To: 刘业; Hara Papadopoulou
Cc: Sales Department 
Subject: Upcoming Shipment
Hello Mary,
This is to inform you about the upcoming shipment.
Thanks and Best regards,
John Doe
Export Manager
Company XYZ Ltd.
190, Venizelos Str 12345 Salamina – Greece
This email has been scanned by XYZ AntiVirus.
***TO AVOID HACKER,PLS RECONFIRM BANK ACCOUNT WITH ME VIA WECHAT OR SKYPE BEFORE ARRANGE PAYMENT***,
Sent from my iphone
                                                         
Output: 
From: johndoe@gmail.com
Sent: Friday, December 29, 2023 14:09 PM
To: 刘业; Hara Papadopoulou
Cc: Sales Department
Subject: Upcoming Shipment
Hello Mary,
This is to inform you about the upcoming shipment.
Thanks and Best regards,
John Doe
End of email                                                
                                                         
Process the following email:
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{email}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id>                                                        
"""
)

headers_cleaning_prompt = PromptTemplate.from_template("""<|start_header_id|>system<|end_header_id|>
You are an email cleaning agent. Your task is to clean the email headers while preserving the email body
exactly. 
                                                       
Rules for cleaning header fields:
1. Process the fields "From:", "To:", "Cc:"  sequentially, using the following rules in order:
    - Each field must contain exactly one identifier per contact, either an email address or a name.
    - If name and email address appear in the same contact, in any order and in any format, keep only the email address.
    - If only the email address appear in the contact, keep only the email address. 
    - If only the name appear in the contact, keep only the name.                                                      
    - Remove apostrophes (" or ') from the contacts.                                            
2. Copy "Subject:" and "Sent:" headers exactly as in the input.

Rules for Body:
- Copy the body text exactly as written.
- Preserve all formatting exactly.

Output Format:
- Start with "From:" and end with newline + "End of email".
- Double-check that all rules are applied and the body text is unchanged.                                                       

Example:
Input: 
From: "John Doe" <johndoe@email.gr> 
Sent: Friday, December 29, 2023 22:00 PM
To: "Mary Joe"; "Hara Papadopoulou" <harapap@gmail.com>; Kate Doe <katedoe@example.com>
Cc: Sales Department 
Subject: Upcoming Shipment
MBL SWB
Hello Mary,
This is to inform you about the upcoming shipment.
Thanks and Best regards
John Doe

Output: 
From: johndoe@email.gr
Sent: Friday, December 29, 2023 22:00 PM
To: Mary Joe; harapap@gmail.com; katedoe@example.com
Cc: Sales Department
Subject: Upcoming Shipment
MBL SWB
Hello Mary,
This is to inform you about the upcoming shipment.
Thanks and Best regards
John Doe
End of email

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
