import json, os, sys

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

#import networkx as nx
from utils.logging_config import LOGGER
from extract_emails_acc import split_and_extract_emails_acc
from tools.build_graph import build_email_graph
from time import time
import torch

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    #graph = nx.DiGraph()
    for filename in os.listdir(dir_path):
        if filename.endswith(".msg"):
            file_path = os.path.join(dir_path, filename)
            try:

                tic = time()
                data = split_and_extract_emails_acc(file_path)
                LOGGER.info(f"Time taken to process {filename}: {time() - tic} seconds")

                #graph = build_email_graph(graph, data, filename)
                email_data.extend(data) 
            except Exception as e:
                LOGGER.error(f"Processing {filename} failed: {e}")
    
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(email_data, file, indent=4, ensure_ascii=False, default=str)
    

