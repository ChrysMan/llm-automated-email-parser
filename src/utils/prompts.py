from langchain_core.prompts import PromptTemplate,FewShotPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings

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

cleaning_prefix = """<|start_header_id|>system<|end_header_id|>
You are an email cleaning agent. You have two tasks:
1. For the fields "From:", "To:", "Cc:":
    - Whatever format you choose for one field must be strictly applied to the rest of the fields, separated by semicolons.
    - If all fields contains both a name and an email address, keep only the email address.
    - If any field contains only a name, keep only the name.
    - If any field contains an email address, keep only the email address.
    - If the name contains apostrophes ("'") remove them.

2. Remove all the information that follow after the **signature name line**. 
    - The signature block starts with phrases like "Best regards", "Thanks and Best regards", "Kind regards", "Sincerely", "Yours faithfully", "Ευχαριστώ", "Ευχαριστώ πολύ", "Με εκτίμηση" or something similar.
    - After this phrase, keep only the very next line if it contains the sender’s name.
    - Delete everything that appears after that name line — including phone numbers, job titles, company names, addresses, disclaimers, antivirus messages, or blank lines.

---Important Rules---
* *DO NOT* hallucinate or add any information that is not present in the email.
* Output only the email text without any additional formatting or explanations. 
* The output should start with "From:" and end with the name of the sender followed by the phrase "---End of email---" once *strictly*.
"""

# Few shot examples
cleaning_examples = [
    {
        "input": """From: John Doe <jdoe@email.com <mailto:jdoe@email.com>>
To: "Mary Joe" 
Cc: Sales Department 
Subject: Upcoming Shipment
Hello Mary,
This is to inform you about the upcoming shipment.
Best regards
John Doe
Export Manager
Company XYZ Ltd.
This email has been scanned by XYZ AntiVirus.""",
        "output": """From: John Doe
To: Mary Joe
Cc: Sales Department
Subject: Upcoming Shipment
Hello Mary,
This is to inform you about the upcoming shipment.
Best regards
John Doe 
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
23B Oxford street | London-England 64336""",
        "output": """Στις Τρίτη, Μαΐου 30, 2023, 11:23 πμ, ο χρήστης mdoe@email.com έγραψε:
Καλησπερα Κε Πετρόπουλε,
Πως ειστε ?
Λαβαμε ενημερωση για ένα νέο φορτιο.
Best regards
Maria Doe (Mrs.)
---End of email---"""
    },
    {
        "input": """From: John Doe 
To: "Mary Joe" <mjoe@gmail.com>; 'Hara Papadopoulou' <harapap@gmail.com>
Cc: <sales@department.com>
Subject: Upcoming Shipment
Hello Mary,
This is to inform you about the upcoming shipment.
Best regards
John Doe
Export Manager
Company XYZ Ltd.
This email has been scanned by XYZ AntiVirus.""",
        "output": """From: John Doe
To: <mjoe@gmail.com>, <harapap@gmail.com>
Cc: <sales@department.com>
Subject: Upcoming Shipment
Hello Mary,
This is to inform you about the upcoming shipment.
Best regards
John Doe 
---End of email---"""        
    }    
]

def create_FewShotPrompt(examples: list, prefix: str):
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        OpenAIEmbeddings(),
        Chroma,
        k=2,
    )

    return FewShotPromptTemplate(
        examples_selector=example_selector,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix="""Process the following email:\n{email}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id>""",
        input_variables=["email"]
    )

