import json, os, sys
import networkx as nx
from utils.logging_config import LOGGER
from chains.extract_emails import split_and_extract_emails_sync
#from tools.build_graph import build_email_graph
from time import time

if __name__ == "__main__":
    if len(sys.argv) != 2:
        LOGGER.error("Usage: python emailParsing.py <dir_path>")
        sys.exit(1)
    
    dir_path = sys.argv[1]

    if not os.path.isdir(dir_path):
        LOGGER.error(f"{dir_path} is not a valid directory.")
        sys.exit(1)

    folder_name = os.path.basename(os.path.normpath(dir_path))

    output_path = os.path.join(dir_path, f"{folder_name}.json")

    email_data = []
    graph = nx.DiGraph()
    for filename in os.listdir(dir_path):
        if filename.endswith(".msg"):
            file_path = os.path.join(dir_path, filename)
            try:

                tic = time()
                data = split_and_extract_emails_sync(file_path)
                LOGGER.info(f"Time taken to process {filename}: {time() - tic} seconds")

                #graph = build_email_graph(graph, data, filename)
                email_data.extend(data) 
            except Exception as e:
                LOGGER.error(f"Processing {filename} failed: {e}")
    
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(email_data, file, indent=4, ensure_ascii=False, default=str)

    LOGGER.info(f"Time taken to process: {time() - tic} seconds")
    

