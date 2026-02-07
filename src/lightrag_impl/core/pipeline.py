import os, json
from time import time
from functools import partial
from lightrag.lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_share_data, initialize_pipeline_status
from typing import Any

from ..core.llm import llm_model_func, rerunk_func
from utils.file_io import read_json_file


WORKING_DIR = "lightrag_impl/core/rag_storage"
# os.makedirs(WORKING_DIR, exist_ok=True)

async def initialize_rag(working_dir: str = WORKING_DIR) -> LightRAG:

    rag = LightRAG(
        working_dir=working_dir,
        graph_storage="Neo4JStorage",
        llm_model_func=llm_model_func, #ollama_model_complete,
        #llm_model_name="qwen2.5:14b", #"Piyush20/Qwen_14B_Quantized", #"qwen2.5:14b", #"llama3.1:8b",
        llm_model_max_async=4,
        #llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
        vector_storage="FaissVectorDBStorage",
        rerank_model_func=rerunk_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=partial(ollama_embed.func, embed_model="bge-m3:latest", host="http://localhost:11434") # Need to pull it first
        )
    )

    # Initialize database connections
    await rag.initialize_storages()
    # Ensure shared dicts exist
    initialize_share_data()

    await initialize_pipeline_status()

    return rag

async def index_data(rag: LightRAG, dir_path: str)-> str:
    """Indexes data from the given file path into the RAG system."""
    if not os.path.isdir(dir_path):
        return f"Error: {dir_path} is not a valid directory."
    elif os.path.isdir(dir_path) and not any(f.endswith("unique.json") for f in os.listdir(dir_path)):
        return f"Error: No 'unique.json' files found in {dir_path}. The .msg files must first be preprocessed."
    

    for filename in os.listdir(dir_path):
            if filename.endswith("unique.json"):
                file_path = os.path.abspath(os.path.join(dir_path, filename))

                email_list = read_json_file(file_path)
                list_of_texts = [json.dumps(j) for j in email_list]
                file_paths= [file_path for _ in list_of_texts]
                
                tic = time()
                await rag.ainsert(input=list_of_texts, file_paths=file_paths)
                
                return f"Indexing of data from {file_path} completed at {time()-tic} seconds."


    # if file_path.endswith(".pdf"):
    #     with pdfplumber.open(file_path) as pdf:
    #         text = "\n".join(page.extract_text() for page in pdf.pages)
    #     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    #     texts = text_splitter.split_text(text)
    #     await rag.ainsert(texts)
    # else:
    #     LOGGER.warning(f"Unsupported file format for {file_path}")
    # deepseek modelo


async def run_async_query(rag: LightRAG, question: str, mode: str) -> dict[str, Any]:
    """
    Execute an async RAG query using .aquery method 
    """
    response = await rag.aquery_llm(
        query=question,
        param=QueryParam(
            mode=mode, 
            enable_rerank=True, 
            include_references=True,
            user_prompt="""Ensure the response remains exclusively focused on the specific question asked; 
            avoid providing peripheral context, unsolicited advice, or any information not directly required to answer the query."""

        )
    )
    # Used for evaluation
    if 'data' in response:
        for d in response.get('data', {}).get('entities', []):
            d.pop("source_id", None)
            d.pop("created_at", None)
        for d in response.get('data', {}).get('relationships', []):
            d.pop("source_id", None)
            d.pop("created_at", None)

    print(f"[_build_query_context] Raw data entities: {response.get('data', {}).get('entities', [])}, relationships: {response.get('data', {}).get('relationships', [])}, chunks: {response.get('data', {}).get('chunks', [])}")

    return response

        
