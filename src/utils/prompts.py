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
Body: Goodmorning Mr Papadopoulos,
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
You are an email translation agent. You will be inputted an email that is part English and part another language. Your job is to keep the email **exactly the same**, except that Greek and Chinese or other language (except English) text segments must be translated into English.
- Copy all English text exactly as it is. Do not paraphrase, rewrite, or invent content.
- Only replace the non-English parts (Greek or Chinese or other) with their English translation in the same position.
- Preserve all headers, punctuation, spacing, line breaks, and formatting exactly.
- Output must start with "From:" and end with "\n---End of email---".
                                                            
Process the following email:
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{email}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id>""" 
)

translator_formatting_prompt2 = """Your are an email translator and formatting agent. Your task is to translate accurately the email into English (strictly)
and format the email.

Rules:
1. Translate the body text in English exactly as it is written. Do not remove or add spaces, do not merge lines, do not change punctuation, and do not alter line breaks.
2. Format the email headers to the following fields:
"From: " + The sender of the email.
"Sent: " + The Date of the email in the format: Full weekday name, full month name day, four-digit year, hour:minute:second AM/PM. Be carefull to not change the date or time.
"To: " + The receivers of the email, if they exist.
"Cc: " + The Cc of the email, if they exist.
"Subject: " + The subject of the email, if it exists.
"Body: " + The body of the email, if it exists.
                                                                                                     
3. Output must start with "From:".

Example 1:
Input:
发件人:  约翰·多伊 <jdoe@email.com >
发送日期: 2023-12-29 14:09:28
收件人: Mary Joe; harapap@gmail.com
主题: Upcoming Shipment
Goodmorning Mr Papadopoulos,
We received an update about a new shipment.
Best regards
John Doe 
提单一律寄顺丰到付，开票只能开0税率电子发票，均不能抵扣.

Output:
From:  John Doe <jdoe@email.com >
Sent: 2023-12-29 14:09:28
To: Mary Joe; harapap@gmail.com
Cc:
Subject: Upcoming Shipment
Body: Goodmorning Mr Papadopoulos,
We received an update about a new shipment.
Best regards
John Doe 
All bills of lading must be sent by SF Express to be paid on delivery. Only 0-tax electronic invoices can be issued for invoicing, and no deductions are allowed.

Example 2: 
Input:
Στις 30/05/2023 9:52 π.μ., ο/η Mairy Doe έγραψε:
καλημέρα σας 
επισυναπτόμενη η φορτωτική προς έλεγχο και επιβεβαίωση. 
Kind regards
Mairy Doe (Ms.)
Export Manager
Company XYZ Ltd.

Output:                                              
From: mdoe@gmail.com
Sent: Tuesday, May 30, 2023 11:23 AM
To: 
Cc:
Subject:
Body: Good morning 
attached is the bill of lading for your review and confirmation.
Kind regards
Mairy Doe (Ms.)
Export Manager
Company XYZ Ltd.
"""

formatting_headers_prompt = PromptTemplate.from_template("""<|start_header_id|>system<|end_header_id|>
You are an email formatting agent. Your task is to reformat the input email into a structured form without altering any content except for the "Sent" date, which must be reformatted.

Rules:
1. Reformat the email headers to the following fields:
"From: " + The sender of the email.
"Sent: " + The Date of the email reformatted into: Full weekday name, full month name day, four-digit year, hour:minute:second AM/PM. Be carefull to not change the date or time.
"To: " + The recipients of the email (if they exist).
"Cc: " + The Cc of the email, (if they exist).
"Subject: " + The subject of the email, (if it exists).
"\n" + The entire body of the email exactly as it appears in the input.

2. You are allowed to translate any text that is not in English, into English.
3. Other than translating and reformatting the "Sent" date, do not modify the body text in any way. Copy it exactly as it is written. 
4. Output must start with "From:" and end with "\n---End of email---".
                                               
Example:
Input: 
On May 30, 2023, at 11:23, Maria Doe <mdoe@email.com> wrote:
Goodmorning Mr Papadopoulos,
We received an update about a new shipment.
Best regards
Maria Doe (Mrs.)
Export Manager
Company XYZ Ltd.

Output: 
From: mdoe@gmail.com
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

Process the following email:
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{email}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id>  
""")


headers_cleaning_prompt = PromptTemplate.from_template("""<|start_header_id|>system<|end_header_id|>
You are an email cleaning agent. Your task is to clean email headers while preserving the body.

For the fields "From:", "To:", "Cc:":
1. Each field must contain exactly one identifier per person.
2. If email address or name are present in a field, keep only the email address.
3. If the name is present, keep it as is.
4. If the email address is present, keep it as is.
5. Do not move information between the fields.
6. If the name contains apostrophes ("),(') remove them.
    
Important Rules:
- Copy the body text exactly as it is written. Do not remove or add spaces, do not merge lines, do not change punctuation, and do not alter line breaks. Preserve all formatting exactly. 
- Output must start with "From:" and end with sender name + "\n---End of email---".

Example:
Input: 
From: John Doe <jdoe@email.com >
Sent: Friday, December 29, 2023 10:00 PM
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

Process the following email:
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{email}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id>  
""")

signature_cleaning_prompt = PromptTemplate.from_template("""<|start_header_id|>system<|end_header_id|>
You are an email cleaning agent. You have one task:
                                                         
Remove all the information that follow after the **signature name line**:
1. The signature block starts with phrases like "Best regards", "Thanks and Best regards", "Kind regards", "Sincerely", "Yours faithfully", "Ευχαριστώ", "Ευχαριστώ πολύ", "Με εκτίμηση" or something similar.
2. After this phrase, keep only the senders' name line.
3. Delete everything that appears after that name line, including phone numbers, job titles, company names, addresses, disclaimers, antivirus messages, footers, device signatures or blank lines.
4. If there is no name signature delete from the end, text that seems irrelevant to the email body like disclaimers, antivirus messages, footers, device signatures or blank lines
                                                  
Important Rules:
- Copy the body text exactly as it is written. Do not remove or add spaces, do not merge lines, do not change punctuation, and do not alter line breaks. Preserve all formatting exactly. 
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
