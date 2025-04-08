from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
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

model = OllamaLLM(model="llama3.1", temperature=0,  num_gpu_layers=50, num_ctx=16384, num_predict=16384) #8192
parser = JsonOutputParser(pydantic_object=EmailContent)
safe_parser = OutputFixingParser.from_llm(parser=parser, llm=model)

prompt = PromptTemplate.from_template(
"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert assistant who processes email threads with precision and no loss of content.
The input consists of several email strings. Some emails may be duplicates. Your task is:
1. Split the email thread into individual emails using "***" as the only delimiter. Each "***" marks the end of one complete email.
2. Identify and remove any duplicate emails in the list based on content mainting the chronological order.
3. Return the unique emails **ONLY** as a JSON list, where each email is a **separate object** with the following fields:

- sender: The sender of the email. Store their name and email, if available, as a string in the format "Name <email@example.com>". If only a name is present, store it as "Name". 
- sent: Extract and include **only the date and time**. 
- to: A list of recipients' names and emails, if available. Store each entry as a string in the format "Name <email@example.com>" or "Name". The "To" field may contain multiple recipients separated by commas or semicolons.
- cc: A list of additional recipients' names and emails. Store each entry as a string in the format "Name <email@example.com>" or "Name". The "Cc" field may contain multiple recipients separated by commas or semicolons.
- subject: The subject of the email, stored as a string. Extract this after the "Subject:" field.
- body:  Include the full content of the message starting from the line **after** "Subject:" or "wrote:", and continue **until the next delimiter**. Do not summarize or skip any content.

4. Maintain the chronological order of the emails in the output.
5. Do not hallucinate or add any information that is not present in the email thread.
6. Output only a raw JSON array. No comments, no markdown, no code block. No natural language."

Process the following email thread:
<|eot_id|><|start_header_id|>user<|end_header_id|>
{emails}
<|eot_id|><|start_header_id|>assistant<|end_header_id>
"""
)

SPLIT_EMAILS_CHAIN = (prompt | model | parser)
