import os
from dotenv import load_dotenv
load_dotenv()

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

qa_llm = ChatOllama(
    model="qwen2.5:14b",
    temperature=0.2
)

cypher_llm = ChatOllama(
    model="qwen2.5:14b",    
    temperature=0
)

# Create the Embedding model
from langchain_ollama import OllamaEmbeddings

#embed_model = "mxbai-embed-large"
embed_model = "nomic-embed-text"

embedding_provider = OllamaEmbeddings(
    model=embed_model
)
