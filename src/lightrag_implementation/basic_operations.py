import os, sys, asyncio, pdfplumber, json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.logging_config import LOGGER
from utils.graph_utils import read_json_file
from lightrag.lightrag import LightRAG
from functools import partial
from lightrag.rerank import cohere_rerank, generic_rerank_api, jina_rerank
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_share_data, initialize_pipeline_status
from langchain_community.document_loaders import DirectoryLoader, TextLoader, JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from typing import Any, Dict

WORKING_DIR = "./lightrag_implementation/rag_storage"
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

rerunk_func = partial(
    generic_rerank_api, 
    model=os.getenv("RERANK_MODEL"),
    base_url=os.getenv("RERANK_BINDING_HOST"),
    api_key=os.getenv("RERANK_BINDING_API_KEY")
)

async def initialize_rag(working_dir: str = WORKING_DIR) -> LightRAG:

    rag = LightRAG(
        working_dir=working_dir,
        graph_storage="Neo4JStorage",
        llm_model_func=ollama_model_complete,
        llm_model_name="llama3.1:8b",#"qwen2.5:14b",
        llm_model_max_async=4,
        llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
        vector_storage="FaissVectorDBStorage",
        rerank_model_func=rerunk_func,
        min_rerank_score=0.5,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts, embed_model="nomic-embed-text", host="http://localhost:11434"
                )
        ),
        enable_llm_cache=False
    )

    # Initialize database connections
    await rag.initialize_storages()
    # Ensure shared dicts exist
    initialize_share_data()
    # Initialize pipeline status for document processing
    await initialize_pipeline_status()

    return rag

async def index_data(rag: LightRAG, dir_path: str):
    """Indexes data from the given file path into the RAG system."""
    for filename in os.listdir(dir_path):
            if filename.endswith("unique.json"):
                file_path = os.path.abspath(os.path.join(dir_path, filename))

                email_list = read_json_file(file_path)
                list_of_texts = [json.dumps(j) for j in email_list]
                file_paths= [file_path for _ in list_of_texts]

                await rag.ainsert(input=list_of_texts, file_paths=file_paths)


    # if file_path.endswith(".pdf"):
    #     with pdfplumber.open(file_path) as pdf:
    #         text = "\n".join(page.extract_text() for page in pdf.pages)
    #     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    #     texts = text_splitter.split_text(text)
    #     await rag.ainsert(texts)
    # else:
    #     LOGGER.warning(f"Unsupported file format for {file_path}")




        
