from langchain_core.prompts import ChatPromptTemplate
#from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel
from chains.email_parser import EmailInfo
from typing import Union, List
'''
class EmailNode(BaseModel):
    id: str
    reply_to: Optional[str] = None
    email: Union[str,EmailInfo] # str after the splitting process and then EmailInfo after extraction process
'''
class EmailList(BaseModel):
    emails: List[Union[str,EmailInfo]] #List[EmailNode]  # str after the splitting process and then EmailInfo after extraction process
    
#parser = PydanticOutputParser(pydantic_object=EmailList)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an expert assistant that extracts individual emails from a conversation thread, \
            arranging them in chronological order (oldest to newest) and extracting them in a list of strings.
            
            The provided text contains an email thread where the first email is the most recent reply, \
            while the last email in the text is the original message that started the conversation. 
            Your task is to extract each email content, ensuring that excessive newlines, spaces, \
            and tab characters at the beginning of each sentence are removed, while preserving the correct reply order.

            ### Formatting Instructions:
            - Eliminate duplicate emails and ensure each email in the output is unique.
            - Ensure emails are seperated accurately, maintaining the integrity of the conversation.      

            Process the following text:
            {text}

            """,
        )
    ]
) #.partial(format_instructions=parser.get_format_instructions())

model = ChatOllama(model="llama3.1", temperature=0)

#SPLIT_EMAILS_CHAIN = (prompt | model | parser)
SPLIT_EMAILS_CHAIN = (prompt | model.with_structured_output(EmailList))
