import os, sys, asyncio, json
from time import time
from lightrag.kg.neo4j_impl import Neo4JStorage 
from utils.logging_config import LOGGER
from utils.graph_utils import read_json_file
from lightrag_implementation.basic_operations import initialize_rag, index_data, run_async_query
from dotenv import load_dotenv

load_dotenv()

def find_file(filename, search_path):
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

async def main()-> None:
# async def main(mode: str, data_path: str)-> None:
    tic = time()
    
    #DOCS_PATH = data_path
    rag = None
    target_file = "kv_store_full_docs.json"
    start_dir = "./"
    file_path = find_file(target_file, start_dir)
    if not file_path:
        LOGGER.error(f"Could not find {target_file} in {start_dir} or its subdirectories.")
        return
    try:
        rag = await initialize_rag()
        dicts = read_json_file(file_path)
        id_list = [json.dumps(d) for d in dicts]
        
        for id in id_list:
            await rag.adelete_by_doc_id(id.replace('"', ''), True)

        #result_message = await index_data(rag, DOCS_PATH)
        # if "Error" in result_message:
        #     LOGGER.error(result_message)
        #     return
        # else:
        #     LOGGER.info(result_message)

        LOGGER.info(f"Total time taken: {time() - tic} seconds")

        # while (q :=input("> ")) != "exit":
        #     toc = time()
        #     resp_async = await run_async_query(rag, q, mode)
        #     print("\n====== Query Result ======\n", resp_async)
        #     LOGGER.info(f"Duration of answering: {time() - toc} seconds\n")
    except Exception as e:
        LOGGER.error(f"An error occured: {e}")
    finally:
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     LOGGER.error("Usage: python create_kg.py <dir_path>")
    #     sys.exit(1)

    # dir_path = sys.argv[1]

    # if not os.path.isdir(dir_path):
    #     LOGGER.error(f"{dir_path} is not a valid dir.")
    #     sys.exit(1)

    #asyncio.run(main(data_path=dir_path, mode="mix"))
    asyncio.run(main())
