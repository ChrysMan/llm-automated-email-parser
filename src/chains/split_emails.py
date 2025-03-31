from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, CommaSeparatedListOutputParser
from langchain_ollama import OllamaLLM
from pydantic import BaseModel, Field
from chains.email_parser import EmailInfo
from typing import Optional, List, TypedDict

class EmailInfo(BaseModel):
    """Data model for email extracted information."""
    sender: str = Field(..., description="The sender of the email. Store their name and email, if available.")
    sent: str = Field(..., description="The date the email was sent.")
    to: Optional[List[str]] = Field(default_factory=list, description="A list of recipients' names and emails.")
    cc: Optional[List[str]] = Field(default_factory=list, description="A list of additional recipients' names and emails.")
    subject: Optional[str] = Field(None, description="The subject of the email.")
    body: str = Field(..., description="The email body, excluding unnecessary whitespace.")


class EmailContent(BaseModel):
    email: List[EmailInfo]

parser = JsonOutputParser()

prompt = PromptTemplate.from_template(
    """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an expert assistant that splits the given email thread into individual emails.
    Return the extracted emails **ONLY** as a JSON list, where each email is a **separate object** 
    with the following fields:
    
    - sender: The sender of the email. Store their name and email, if available, as a string in the format "Name <email@example.com>". If only a name is present, store it as "Name". 
    - sent: The date sent in any of these formats: "Monday, February 6, 2023 3:58 PM" (full weekday, month, 12-hour format, and AM/PM) OR "Feb 6, 2023, at 3:58"
    - to: A list of recipients' names and emails, if available. Store each entry as a string in the format "Name <email@example.com>" or "Name". The "To" field may contain multiple recipients separated by commas or semicolons.
    - cc: A list of additional recipients' names and emails. Store each entry as a string in the format "Name <email@example.com>" or "Name". The "Cc" field may contain multiple recipients separated by commas or semicolons.
    - subject: The subject of the email, stored as a string. Extract this after the "Subject:" field or "wrote:".
    - body: The email body, stored without unnecessary whitespace.

    Process the following email thread:
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    {emails}
    <|eot_id|><|start_header_id|>assistant<|end_header_id>
    """
)#.partial(format_instructions = parser.get_format_instructions())

model = OllamaLLM(model="llama3.1", temperature=0,  num_gpu_layers=50, num_ctx=8192, num_predict=8192) #8192

SPLIT_EMAILS_CHAIN = (prompt | model | parser)
