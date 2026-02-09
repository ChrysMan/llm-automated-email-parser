import os
from dotenv import load_dotenv
load_dotenv()

from pydantic_ai.providers.openai import OpenAIProvider
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings


llm = ChatOpenAI(
    model=os.getenv("LLM_AGENT_MODEL", "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8"),
    base_url=os.getenv("LLM_AGENT_BINDING_HOST"), 
    api_key=os.getenv("LLM_AGENT_BINDING_API_KEY"),
    temperature=0    
)

embedding_provider = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs = {'normalize_embeddings': True}  
)


