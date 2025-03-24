from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel
from chains.email_parser import EmailInfo
from typing import Union, List, TypedDict

class EmailList(TypedDict):
    emails: List[str] 
    
#parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an expert assistant that splits the given email thread into individual emails.
            Return each email's full content (including header and body) as a separate string in a list format.
            Each email should be placed on a new line in the list, and the full email content should be returned.
            
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
            [
                "From: John Doe <john@example.com>
                Sent: Tuesday, March 10, 2025 9:00 AM
                Subject: Meeting Reminder
                Hello, just a reminder about the meeting tomorrow. ",
                "From: Jane Doe <jane@example.com>
                Sent: Tuesday, March 10, 2025 9:30 AM
                Subject: Re: Meeting Reminder
                Thanks for the reminder! I will be there."
            ]

            The above example is how you should format the email thread into a list of strings. 
            Make sure each email is treated as a separate entry in the list.

            Process the following email thread:
            {emails}
            """,
        )
    ]
)#.partial(format_instructions=parser.get_format_instructions())

model = ChatOllama(model="llama3.1", temperature=0)

#SPLIT_EMAILS_CHAIN = (prompt | model.with_structured_output(EmailList) | parser)
SPLIT_EMAILS_CHAIN = (prompt | model.with_structured_output(EmailList))
