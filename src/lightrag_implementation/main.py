import os, sys, asyncio
from time import time
from logging_config import LOGGER
from basic_operations import initialize_rag, index_data
from lightrag_implementation.retrieve import run_async_query
from dotenv import load_dotenv

load_dotenv()
"""
1. Initialize RAG system
2. Index data from specified directory
3. Run async queries against the RAG system
"""

async def main(mode: str, data_path: str)-> None:
    tic = time()
    
    DOCS_PATH = data_path
    rag = None
    try:
        rag = await initialize_rag()
        await index_data(rag, DOCS_PATH)

        LOGGER.info(f"Total time taken: {time() - tic} seconds")

        while (q :=input("> ")) != "exit":
            toc = time()
            resp_async = await run_async_query(rag, q, mode)
            print("\n====== Query Result ======\n", resp_async)
            LOGGER.info(f"Duration of answering: {toc} seconds\n")
    except Exception as e:
        LOGGER.error(f"An error occured: {e}")
    finally:
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        LOGGER.error("Usage: python create_kg.py <dir_path>")
        sys.exit(1)

    dir_path = sys.argv[1]

    if not os.path.isdir(dir_path):
        LOGGER.error(f"{dir_path} is not a valid dir.")
        sys.exit(1)

    asyncio.run(main(data_path=dir_path, mode="mix"))
