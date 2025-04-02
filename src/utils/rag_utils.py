from langchain_community.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings

sample_emails = [
    {"format": "From: [sender] Sent: [sent] To: [to] Cc: [cc] Subject: [subject] [body]",
    "example": """From: Jane Doe <jane@example.com>
    Sent: Monday, February 20, 2023 02:42 PM
    To: George Doe <george@example.com> 
    Cc: Maria Doe <Maria@example.com>; Kate Doe <Kate@example.com> 
    Subject: Meeting Reminder
    Dear all, please remember to attend the meeting tomorrow at 10 AM.
    Best regards, 
    Jane Doe"""},
    {"format": "On [sent], [sender] wrote: [body]",
    "example": """On February 20, 2023, at 13:23, Jane Doe <jane@example.com> wrote:"
    Dear all, please remember to attend the meeting tomorrow at 10 AM.
    Best regards, 
    Jane Doe"""}
]

vectorstore = Chroma(persist_directory=="./email_formats")