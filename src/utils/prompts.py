from langchain_core.prompts import PromptTemplate,FewShotPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_huggingface import HuggingFaceEmbeddings

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

headers_cleaning_prefix = """<|start_header_id|>system<|end_header_id|>
You are an email cleaning agent. You have one tasks:
For the fields "From:", "To:", "Cc:":
    - If any fields contain both a name and an email address, always prefer to keep the email address.
    - The general preference is to keep the email address, but if it is not available, keep the name.
    - If the name contains apostrophes ("'") remove them.
    

---Important Rules---
* *DO NOT* hallucinate or add any information that is not present in the email.
* Output only the email text without any additional formatting or explanations. 
* The output should start with "From:" and end with the name of the sender followed by the phrase "---End of email---" once *strictly*.
"""

signature_cleaning_prompt = PromptTemplate.from_template("""<|start_header_id|>system<|end_header_id|>
You are an email cleaning agent. You have one tasks:
Remove all the information that follow after the **signature name line**. 
    - The signature block starts with phrases like "Best regards", "Thanks and Best regards", "Kind regards", "Sincerely", "Yours faithfully", "Ευχαριστώ", "Ευχαριστώ πολύ", "Με εκτίμηση" or something similar.
    - After this phrase, keep only the very next line if it contains the sender’s name.
    - Delete everything that appears after that name line, including phone numbers, job titles, company names, addresses, disclaimers, antivirus messages, or blank lines.

---Important Rules---
* *DO NOT* hallucinate or add any information that is not present in the email.
* Output only the email text without any additional formatting or explanations. 
* The output should start with "From:" and end with the name of the sender followed by the phrase "---End of email---" once *strictly*.

Example:
    "input": From: John Doe 
Date: Friday, December 29, 2023 2:09 PM
To: 刘业; Hara Papadopoulou
Cc: Sales Department 
Subject: Upcoming Shipment
Hello Mary,
This is to inform you about the upcoming shipment.
Thanks and Best regards
John Doe
Export Manager
Company XYZ Ltd.
This email has been scanned by XYZ AntiVirus.
***TO AVOID HACKER,PLS RECONFIRM BANK ACCOUNT WITH ME VIA WECHAT OR SKYPE BEFORE ARRANGE PAYMENT***,
    "output": From: John Doe 
Sent: Friday, December 29, 2023 2:09 PM
To: 刘业; Hara Papadopoulou
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
    },
    {
        "input": """From: John Doe <jdoe@email.com >
Sent: Friday, December 29, 2023 2:09 PM
To: "Mary Joe"  <jdoe@email.com>; "Hara Papadopoulou" <harapap@gmail.com>
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
To: jdoe@email.com; harapap@gmail.com
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
    },
    {
        "input": """Στις Τρίτη, Μαΐου 30, 2023, 11:23 πμ, ο χρήστης Maria Doe <mdoe@email.com> έγραψε:
Καλησπερα Κε Πετρόπουλε,
Πως ειστε ?
Λαβαμε ενημερωση για ένα νέο φορτιο.
Best regards
Maria Doe (Mrs.)
Operation manager
Company Name S.A.
23B Oxford street | London-England 64336
***TO AVOID HACKER,PLS RECONFIRM BANK ACCOUNT WITH ME VIA WECHAT OR SKYPE BEFORE ARRANGE PAYMENT***""",
        "output": """From: mdoe@gmail.com
Sent: Τρίτη, Μαίου 30, 2023 11:23 AM
To: Κε Πετρόπουλε
Cc:
Subject:
Καλησπερα Κε Πετρόπουλε,
Πως ειστε ?
Λαβαμε ενημερωση για ένα νέο φορτιο.
Best regards
Maria Doe (Mrs.)
Operation manager
Company Name S.A.
23B Oxford street | London-England 64336
***TO AVOID HACKER,PLS RECONFIRM BANK ACCOUNT WITH ME VIA WECHAT OR SKYPE BEFORE ARRANGE PAYMENT***
---End of email---"""
    },
    {
        "input": """From: John Doe 
Sent: Friday, December 29, 2023 2:09 PM
To: "Mary Joe"; 'Hara Papadopoulou' <harapap@gmail.com>
Cc: <sales@department.com>
Subject: Upcoming Shipment
Hello Mary,
This is to inform you about the upcoming shipment.
Thanks and Best regards
John Doe
Export Manager
Company XYZ Ltd.
This email has been scanned by XYZ AntiVirus.
***TO AVOID HACKER,PLS RECONFIRM BANK ACCOUNT WITH ME VIA WECHAT OR SKYPE BEFORE ARRANGE PAYMENT***""",
        "output": """From: John Doe
Sent: Friday, December 29, 2023 2:09 PM
To: Mary Joe, <harapap@gmail.com>
Cc: <sales@department.com>
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
        suffix="""Process the following email:\n{email}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id>""",
        input_variables=["email"]
    )

