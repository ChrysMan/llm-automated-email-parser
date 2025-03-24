from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import List, Optional

class EmailInfo(BaseModel):
    """Data model for email extracted information."""
    sender: str = Field(..., description="The sender of the email. Store their name and email, if available.")
    sent: str = Field(..., description="The date the email was sent.")
    to: Optional[List[str]] = Field(default_factory=list, description="A list of recipients' names and emails.")
    cc: Optional[List[str]] = Field(default_factory=list, description="A list of additional recipients' names and emails.")
    subject: Optional[str] = Field(None, description="The subject of the email.")
    body: str = Field(..., description="The email body, excluding unnecessary whitespace.")
 
parser = JsonOutputParser(pydantic_object=EmailInfo)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an expert assitant for extracting metadata from email in JSON format.  
            From the given email extract the following information, according to provided JSON schema: \n

            - sender: The sender of the email. Store their name and email, if available, as a string in the format "Name <email@example.com>". If only a name is present, store it as "Name". 
            - sent: The date sent in any of these formats: "Monday, February 6, 2023 3:58 PM" (full weekday, month, 12-hour format, and AM/PM) OR "Feb 6, 2023, at 3:58"
            - to: A list of recipients' names and emails, if available. Store each entry as a string in the format "Name <email@example.com>" or "Name". The "To" field may contain multiple recipients separated by commas or semicolons.
            - cc: A list of additional recipients' names and emails. Store each entry as a string in the format "Name <email@example.com>" or "Name". The "Cc" field may contain multiple recipients separated by commas or semicolons.
            - subject: The subject of the email, stored as a string. Extract this after the "Subject:" field or "wrote:".
            - body: The email body, stored without unnecessary whitespace.


            ### Instructions:  
            - Only extract from the provided email text. Do not hallucinate.
            - **DO NOT include data from the example.** 
            - **Output only valid JSON.** Ensure you extract all fields, even if they appear on a separate line or have extra spaces.

            ### Example (do not extract from this):
            json
            {{
                "sender": "<sender_name> <<sender_email>>",
                "sent": "<day>, <month> <date>, <year> <time> <AM/PM>",
                "to": ["<recipient1_name> <<recipient1_email>>", "<recipient2_name>"],
                "cc": ["<cc1_name> <<cc1_email>>", "<cc2_name>", "<cc3_name> <<cc3_email>>"],
                "subject": "<email_subject>",
                "body": "<email_body_text>"
            }}

            Process the following email:
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            {email}
            <|eot_id|><|start_header_id|>assistant<|end_header_id>
            """,
        )
    ]
).partial(format_instructions=parser.get_format_instructions())

model = ChatOllama(model="llama3.1", temperature=0)

EMAIL_PARSER_CHAIN = (prompt | model | parser)

    