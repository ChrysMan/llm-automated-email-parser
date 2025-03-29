from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, CommaSeparatedListOutputParser
from langchain_ollama import OllamaLLM
from pydantic import BaseModel, Field
from chains.email_parser import EmailInfo
from typing import Union, List, TypedDict

class EmailContent(BaseModel):
    email: List[str]= Field(description="list of full raw email content")

parser = JsonOutputParser(pydantic_object=EmailContent)#CommaSeparatedListOutputParser()

prompt = PromptTemplate.from_template(
    """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an expert assistant that splits the given email thread into individual emails.
    Only return each email's full raw content (including header and body) as a separate string in a list format in a JSON.
    Do NOT provide any summaries, explanations, or additional text.
    Each email should be placed on a new line in the list, and the full email content should be returned.
    
    {format_instructions}
    
    **Example Email Thread**:
    ```
    From: John Doe <john@example.com>
    Sent: Tuesday, March 10, 2025 9:00 AM
    Subject: Meeting Reminder

    Hello, just a reminder about the meeting tomorrow. 

    From: Jane Doe <jane@example.com>
    Sent: Tuesday, March 10, 2025 9:30 AM
    Subject: Re: Meeting Reminder
    
    Thanks for the reminder! I will be there.
    ```

    **Expected Output Format**:
    json
    {{
        "emails: [
            "From: John Doe <john@example.com>
            Sent: Tuesday, March 10, 2025 9:00 AM
            Subject: Meeting Reminder
            Hello, just a reminder about the meeting tomorrow. ",
            "From: Jane Doe <jane@example.com>
            Sent: Tuesday, March 10, 2025 9:30 AM
            Subject: Re: Meeting Reminder
            Thanks for the reminder! I will be there."
        ]
    }}

    Process the following email thread:
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    {emails}
    <|eot_id|><|start_header_id|>assistant<|end_header_id>
    """
).partial(format_instructions = parser.get_format_instructions())

model = OllamaLLM(model="llama3.1", temperature=0,  num_gpu_layers=-1, num_ctx=131072, num_predict=8192)

#SPLIT_EMAILS_CHAIN = (prompt | model.with_structured_output(EmailList) | parser)
SPLIT_EMAILS_CHAIN = (prompt | model)
