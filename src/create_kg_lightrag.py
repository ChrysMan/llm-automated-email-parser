import os, sys, asyncio, pdfplumber
from utils.logging_config import LOGGER
from lightrag.lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from langchain_community.document_loaders import DirectoryLoader, TextLoader, JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from neo4j import GraphDatabase


WORKING_DIR = "./lightrag_data"
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

NEO4J_URI=os.getenv('NEO4J_URI')
NEO4J_USERNAME=os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD=os.getenv('NEO4J_PASSWORD')

async def initialize_lightrag() -> LightRAG:

    rag = LightRAG(
        working_dir=WORKING_DIR,
        graph_storage="Neo4JStorage",
        llm_model_name="llama3.1:8b",
        llm_model_max_async=4,
        # rerank_model_funk=
        # min_rerank_score=
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
    # Initialize pipeline status for document processing
    await rag.initialize_pipeline_status()

    return rag

async def main():
    if len(sys.argv) != 2:
        LOGGER.error("Usage: python create_kg.py <dir_path>")
        sys.exit(1)

    dir_path = sys.argv[1]

    if not os.path.isdir(dir_path):
        LOGGER.error(f"{dir_path} is not a valid dir.")
        sys.exit(1)

    
    DOCS_PATH = dir_path
    try:
        rag = await initialize_lightrag()
        json_loader = DirectoryLoader(DOCS_PATH, glob="*unique.json", loader_cls=lambda path: JSONLoader(path, jq_schema=".[]", text_content=False), show_progress=True)
        json_docs = json_loader.load()
        json_list = [doc.page_content for doc in json_docs]
        await rag.ainsert(json_list)  

    except Exception as e:
        LOGGER.error(f"An error occured: {e}")
    finally:
        if rag:
            await rag.finalize_storages()

if __name__ == "__main__":
    asyncio.run(main())


        
