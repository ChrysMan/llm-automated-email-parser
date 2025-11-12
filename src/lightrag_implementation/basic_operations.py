import os, sys, asyncio, pdfplumber, json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.logging_config import LOGGER
from utils.graph_utils import read_json_file
from lightrag.lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_share_data, initialize_pipeline_status
from langchain_community.document_loaders import DirectoryLoader, TextLoader, JSONLoader
from langchain.text_splitter import CharacterTextSplitter
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

def bge_rerank_func(query: str, docs: list[str]) -> list[float]:
    """Return reranking scores for each doc given a query."""
    pairs = [[query, doc] for doc in docs]
    device = reranker_model.device
    with torch.no_grad():
        inputs = reranker_tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
        ).to(device)
        scores = reranker_model(**inputs, return_dict=True).logits.view(-1).float()
    return scores.cpu().tolist()


async def initialize_rag(working_dir: str = "./lightrag_implementation/rag_storage") -> LightRAG:

    rag = LightRAG(
        working_dir=working_dir,
        graph_storage="Neo4JStorage",
        llm_model_name="llama3.1:8b",
        llm_model_max_async=4,
        rerank_model_func=bge_rerank_func,
        min_rerank_score=0.5,
        vector_storage="FaissVectorDBStorage",
        llm_model_func=ollama_model_complete,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts, embed_model="nomic-embed-text", host="http://localhost:11434"
                )
        )
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


        
