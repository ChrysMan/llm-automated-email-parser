from langchain_core.prompts import PromptTemplate
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

overall_cleaning_prompt = PromptTemplate.from_template("""<|start_header_id|>system<|end_header_id|>
You are an email cleaning agent. The output will be used in automated downstream parsing, so consistency and strict adherence to rules are critical.
You have to follow the tasks sequencially. The output of every task will be the input to the next task.                                         

Task 1: Translate every non-English segment into proper natural English words. If everything is in English proceed to the next task. 
    Translation Rules:
    1. Do not modify text that is already in English. 
    2. Do not use transliteration. For example "Δευτέρα" becomes "Monday", not "Deutera".  
                                                       
Task 2: Pay **attention** to this step! Remove irrelevant text from the end of the email body.
    Signature cleaning rules:
    1. Identify a signature block: It begins with a closing greeting "Best regards", "Thanks and Best regards", "Kind regards", "Sincerely", "Yours faithfully", "Tks & Best Regards", "Ευχαριστώ", "Ευχαριστώ πολύ", "Με εκτίμηση" or similar) and ends with the sender's name on the next line.  
    2. Delete everything after the sender's name: phone numbers, job titles, company names, disclaimers, antivirus notices, footers, or device signatures.  
    3. If no closing greeting + name is found, remove only irrelevant trailing text (like disclaimers, antivirus notices, footers, device signatures). 
    4. If you find irrelevant text like notes, disclaimers etc before the signature block, remove it with caution, only if it is clearly not part of the email body.
    5. Preserve the body until the cut point.  
                                                                                                              
Task 3: Reformat email headers into a standardized structure while preserving the email text exactly as it appears in the input.
    Formatting Rules: 
    1. Format headers into the following fields (in this exact order):
        - From: + sender (exactly as in input)
        - Sent: + date/time converted into English format -> Full weekday name, full month name day, four-digit year, hour:minute:second AM/PM. Do not change the actual date/time values, only reformat them. "πμ" becomes "AM", "μμ" becomes "PM".
        - To: + recipients (leave blank if not specified)
        - Cc: + recipients (leave blank if not specified)
        - Subject: + subject text (stop reading subject after the newline; leave blank if not specified)
        - In a newline copy the entire body text after the subject line exactly as in the input (preserve all line breaks, spaces, punctuation). 
                                                                           
Task 4: Remove duplicates and apostrophes (" or ') from the contacts in the headers. 
    Headers cleaning rules:
    1. Process the fields "From:", "To:", "Cc:"  sequentially, using the following rules in order:
        - Remove the duplicate emails and names from the contacts.                                                     
        - Remove apostrophes (" or ') from the contacts.                                            
    2. Copy "Subject:" and "Sent:" headers exactly as in the input from the previous task.    

Output format:
   - The output must always start with "From:".
   - End the output with:  
    - If a signature block exists -> sender's name + newline + "End of email"  
    - Otherwise -> newline + "End of email"         

Example 1:
Input: 
On May 30, 2023, at 11:23, Maria Doe wrote:
Goodmorning Mr Manoudakis,
Λάβαμε μια ενημέρωση σχετικά με μια νέα αποστολή στις 12:00 μμ.
Note: The office will be closed on the 28th of October.
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
Goodmorning Mr Manoudakis,
We received an update about a new shipment at 12:00 pm.
Best regards
Maria Doe (Mrs.)
End of email          

Example 2:
Input:                                                      
发件人:  约翰·多伊 
发送日期: 29/12/2023, 22:00 μμ
收件人: 'Nefeli Joe'; Zoi Papa <zoipap@gmail.com>
Cc: Mary Joe/Ms. Mary Joe; harapap@gmail.com <harapap@gmail.com>; "Kate Doe (katedoe@example.com)" <katedoe@example.com> 
主题: Προσφορά για Ντουμπάι
MBL SWB
Goodmorning Mr Papadopoulos,
We received an update about a new shipment.
Kind regards
John Doe 
190, Venizelos Str 12345 Salamina – Greece
This email has been scanned by XYZ AntiVirus.
***TO AVOID HACKER,PLS RECONFIRM BANK ACCOUNT WITH ME VIA WECHAT OR SKYPE BEFORE ARRANGE PAYMENT***,
Sent from my iphone                   

Output:
From:  John Doe 
Sent: Friday, December 29, 2023 22:00 PM
To: Nefeli Joe; Zoi Papa <harapap@gmail.com>
Cc: Mary Joe; harapap@gmail.com; Kate Doe <katedoe@example.com> 
Subject: Offer for Dubai
MBL SWB                                                      
Goodmorning Mr Papadopoulos,
We received an update about a new shipment.
Kind regards
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

formatter_and_translator_prompt = PromptTemplate.from_template("""<|start_header_id|>system<|end_header_id|>
You are an email cleaning agent. Consistency and strict adherence to rules are critical.
You have to follow the tasks sequencially. The output of every task will be the input to the next task.                                         

Task 1: Translate every non-English segment into proper natural English words. If everything is in English proceed to the next task. 
    Translation Rules:
    1. Do not modify text that is already in English. 
    2. Do not use transliteration. For example "Δευτέρα" becomes "Monday", not "Deutera".  
                                                                                                              
Task 2: Reformat email headers into a standardized structure while preserving the email text exactly as it appears in the input.
    Formatting Rules: 
    1. Format headers into the following fields (in this exact order):
        - From: + sender (exactly as in input)
        - Sent: + date/time converted into English format -> Full weekday name, full month name day, four-digit year, hour:minute:second AM/PM. Do not change the actual date/time values, only reformat them. "πμ" becomes "AM", "μμ" becomes "PM".
        - To: + recipients (leave blank if not specified)
        - Cc: + recipients (leave blank if not specified)
        - Subject: + subject text (stop reading subject after the newline; leave blank if not specified)
        - In a newline copy the entire body text after the subject line exactly as in the input (preserve all line breaks, spaces, punctuation).                                                                

Output format:
   - The output must always start with "From:" and end with newline + "End of email".    

Example 1:
Input: 
On May 30, 2023, at 11:23, Maria Doe wrote:
Goodmorning Mr Manoudakis,

Note: The office will be closed on the 28th of October.
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
Goodmorning Mr Manoudakis,
We received an update about a new shipment at 12:00 pm.
Best regards
Maria Doe (Mrs.)
Export Manager
Company XYZ Ltd.    
End of email          

Example 2:
Input:                                                      
发件人:  约翰·多伊 
发送日期: 29/12/2023, 22:00 μμ
收件人: 'Nefeli Joe'; Zoi Papa <zoipap@gmail.com>
Cc: Mary Joe; harapap@gmail.com; Kate Doe <katedoe@example.com
主题: Προσφορά για Ντουμπάι
MBL SWB
Goodmorning Mr Papadopoulos,
We received an update about a new shipment.
Kind regards
John Doe 
190, Venizelos Str 12345 Salamina – Greece
This email has been scanned by XYZ AntiVirus.                

Output:
From:  John Doe 
Sent: Friday, December 29, 2023 22:00 PM
To: 'Nefeli Joe'; Zoi Papa <harapap@gmail.com>
Cc: Mary Joe; harapap@gmail.com; Kate Doe <katedoe@example.com> 
Subject: Offer for Dubai
MBL SWB                                                      
Goodmorning Mr Papadopoulos,
We received an update about a new shipment.
Kind regards
John Doe 
190, Venizelos Str 12345 Salamina – Greece
This email has been scanned by XYZ AntiVirus.
End of email   

Process the following email:
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{email}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id>                                                                                                                                                                   
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
Goodmorning Mr Manoudaki,
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
Goodmorning Mr Manoudaki,
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

cleaning_prompt = PromptTemplate.from_template("""<|start_header_id|>system<|end_header_id|>
You are an email cleaning agent. You have two tasks:
1. Remove irrelevant text from the end of the email body      
2. Remove duplicates and apostrophes (" or ') from the contacts in the headers.     

Rules for task 1:
(1) A signature block begins with a closing greeting "Best regards", "Thanks and Best regards", "Kind regards", "Sincerely", "Yours faithfully", "Tks & Best Regards", "Ευχαριστώ", "Ευχαριστώ πολύ", "Με εκτίμηση" or similar) and ends with the sender's name on the next line.  
(2) AFTER the sender’s name, delete *ONLY* clearly irrelevant signature elements, including:
    - HTML hyperlinks or plain URLs
    - Social media links
    - Legal disclaimers or confidentiality notices
    - Antivirus or security scan notices
    - Marketing slogans or warnings
    - Device or email client signatures
      (e.g., "Sent from my iPhone", "Sent from Outlook")
                                               
(3) Do *NOT* delete the following, even if they appear after the sender’s name or resemble a footer:
    - Job titles or roles
    - Company or organization names
    - Company postal addresses
    - Office location information (street, city, country, postal code)
    - Phone and fax numbers
    - Mobile numbers
                                            
(4) If no closing greeting + name is found, remove only clearly irrelevant trailing content (e.g. disclaimers, antivirus notices, footers, device signatures), and preserve all meaningful email body text.
(5) Do not invent, rewrite, or add content. Preserve all spacing, line breaks, punctuation, and formatting exactly as in the input.  

Rules for task 2:                             
(1) Process the fields "From:", "To:", "Cc:"  sequentially, using the following rules in order:
    - Remove the duplicate emails and names from the contacts.                                                     
    - Remove apostrophes (" or ') from the contacts.                                            
(2) Copy "Subject:" and "Sent:" headers exactly as in the input.
                                               
Output format:  
- Preserve the body until the cut point.  
- End the output with:  
   - If a signature block exists -> sender's name + newline + "End of email"  
   - Otherwise -> newline + "End of email" 
                                               
Example:
Input: 
From: johndoe@email.gr
Sent: Friday, December 29, 2023 22:00 PM
To: Mary Joe/Ms. "Mary Joe"; harapap@gmail.com <harapap@gmail.com>; 'Kate Doe' <katedoe@example.com> 
Cc: Sales Department
Subject: Upcoming Shipment
Hello Mary,
This is to inform you about the upcoming shipment.
Thanks and Best regards,
John Doe
Export Manager
Company XYZ Ltd.
190, Venizelos Str 12345 Salamina – Greece
Office: +30 210 1234567
Mobile: +30 694 1234567
www.companyxyz.gr <http://www.companyxyz.gr/>
This email has been scanned by XYZ AntiVirus.
***TO AVOID HACKER,PLS RECONFIRM BANK ACCOUNT WITH ME VIA WECHAT OR SKYPE BEFORE ARRANGE PAYMENT***,
Sent from my iphone
                                                         
Output: 
From: johndoe@email.gr
Sent: Friday, December 29, 2023 22:00 PM
To: Mary Joe; harapap@gmail.com; Kate Doe <katedoe@example.com> 
Cc: Sales Department
Subject: Upcoming Shipment
Hello Mary,
This is to inform you about the upcoming shipment.
Thanks and Best regards,
John Doe
Export Manager
Company XYZ Ltd.
190, Venizelos Str 12345 Salamina – Greece
End of email                                              
                                                         
Process the following email:
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{email}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id>                                                      
"""
)                 

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
