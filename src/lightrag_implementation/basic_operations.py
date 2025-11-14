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
from dotenv import load_dotenv

load_dotenv()

WORKING_DIR = "./lightrag_implementation/rag_storage"
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

reranker_name = "BAAI/bge-reranker-v2-m3"
reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_name)
reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_name)
reranker_model.eval()
reranker_model.to("cuda" if torch.cuda.is_available() else "cpu")

async def bge_rerank_func(query: str, documents: list[str], top_n: int = None) -> list[Dict[str,Any]]:
    """Return reranking scores for each doc given a query."""
    try:
        pairs = [[query, doc] for doc in documents]
        device = reranker_model.device

        # Run blocking model inference in a thread 
        def run_inference():
            with torch.no_grad():
                inputs = reranker_tokenizer(
                    pairs, padding=True, truncation=True,
                    return_tensors="pt", max_length=512
                ).to(device)
                scores = reranker_model(**inputs, return_dict=True).logits.view(-1).float()
                return scores.cpu().tolist()
        
        # Run CPU-bound inference asynchronously
        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(None, run_inference)

        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        if top_n is not None:
            scored_docs = scored_docs[:top_n]
        
        #print([{"content": document, "relevance_score": score} for document, score in scored_docs])
        return [{"index": document, "relevance_score": score} for document, score in scored_docs]
    except Exception as e:
        LOGGER.error(f"ERROR during reranking: {e}")
        return [
            {"document": document, "relevance_score": 0.0}
            for document in documents
        ]

async def initialize_rag(working_dir: str = "./lightrag_implementation/rag_storage") -> LightRAG:

    rag = LightRAG(
        working_dir=working_dir,
        graph_storage="Neo4JStorage",
        llm_model_func=ollama_model_complete,
        llm_model_name="llama3.1:8b",#"qwen2.5:14b",
        llm_model_max_async=4,
        llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
        vector_storage="FaissVectorDBStorage",
        #rerank_model_func=rerank_model_func,
        #min_rerank_score=0.5,
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
                LOGGER.warning(f"Data file not found: {filename}")

    # if file_path.endswith(".pdf"):
    #     with pdfplumber.open(file_path) as pdf:
    #         text = "\n".join(page.extract_text() for page in pdf.pages)
    #     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    #     texts = text_splitter.split_text(text)
    #     await rag.ainsert(texts)
    # else:
    #     LOGGER.warning(f"Unsupported file format for {file_path}")




        
