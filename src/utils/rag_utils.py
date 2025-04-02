from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings

sample_emails = [
    {"format": "\nFrom: [sender]\nSent:[sent\nTo: [to]\nCc: [cc]\nSubject: [subject]\n[body]",
    "example": """From: Jane Doe <jane@example.com>
    Sent: Monday, February 20, 2023 02:42 PM
    To: George Doe <george@example.com> 
    Cc: Maria Doe <Maria@example.com>; Kate Doe <Kate@example.com> 
    Subject: Meeting Reminder
    Dear all, please remember to attend the meeting tomorrow at 10 AM.
    Best regards, 
    Jane Doe"""},
    {"format": "\nOn [sent], [sender] wrote:\n[body]",
    "example": """On February 20, 2023, at 13:23, Jane Doe <jane@example.com> wrote:
    Dear all, please remember to attend the meeting tomorrow at 10 AM.
    Best regards, 
    Jane Doe"""}
]

vectorstore = Chroma(persist_directory="./email_formats", embedding_function=OllamaEmbeddings(model="llama3.1"), collection_name="email_formats")

for email in sample_emails:
    vectorstore.add_texts(texts=[email["example"]], metadatas=[{"format": email["format"]}])

##
def retrieve_similar_email_format(email_text):
    """Retrieve the most similar email format from the vector DB"""
    results = vectorstore.similarity_search(email_text, k=1)
    if results:
        email_format = results[0].metadata.get("format") # or similar_emails[0].page_content
    
        if email_format:
            return email_format
        else:
            return None
    else:
        return None