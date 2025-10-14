import os
from dotenv import load_dotenv
load_dotenv()

# tag::llm[]
# Create the LLM
from langchain_ollama import ChatOllama

# model = "mistral-7b"
model = "neural-chat:latest"
# model = "gemma3:latest"
# model = "llama3.2:3b"
# model = "llama3.1:8b"

llm = ChatOllama( 
    model=model,
    temperature=0
)
# end::llm[]

# tag::embedding[]
# Create the Embedding model
from langchain_ollama import OllamaEmbeddings

#embed_model = "mxbai-embed-large"
embed_model = "nomic-embed-text"

embedding_provider = OllamaEmbeddings(
    model=embed_model
)
