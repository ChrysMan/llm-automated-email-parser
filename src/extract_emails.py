import extract_msg, json, re, os, sys, ollama
import networkx as nx
from networkx.readwrite import json_graph
from typing import List
from utils.logging_config import LOGGER
from utils.graph_utils import extract_msg_file, append_file, chunk_emails, clean_data, split_email_thread
from chains.split_emails import SPLIT_EMAILS_CHAIN 

def split_emails(file_path: str) -> List[str]:
    raw_msg_content = extract_msg_file(file_path)
    cleaned_msg_content = clean_data(raw_msg_content)

    LOGGER.info("Splitting emails...")
    try:
        raw_model_output = []
        splitted_emails = split_email_thread(cleaned_msg_content)
        reversed_list = splitted_emails[::-1]
        for chunk in chunk_emails(reversed_list, chunk_size=6):
            chunk = "\n*** \n".join(chunk)
            append_file(chunk, "emailSplit.txt")
    
            raw_model_output.extend(SPLIT_EMAILS_CHAIN.invoke({"emails": chunk}))
        LOGGER.info("Splitted emails...")        
    
    except Exception as e:
        LOGGER.error(f"Failed to split emails: {e}")

    graph = nx.DiGraph()

    LOGGER.info("Creating Graph...")
    try:
        for n, email in enumerate(raw_model_output):
            n+=1
            graph.add_node(n, email_node=email)
            if n > 1: 
                graph.add_edge(n-1, n)
    except Exception as e:
        LOGGER.error(f"Failed to create graph: {e}")


    return raw_model_output

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
    for filename in os.listdir(dir_path):
        if filename.endswith(".msg"):
            file_path = os.path.join(dir_path, filename)
            file_path = os.path.abspath(os.path.join(dir_path, filename))
            try:               
                email_data.extend(split_emails(file_path)) 
            except Exception as e:
                LOGGER.error(f"Processing {filename} failed: {e}")
    
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(email_data, file, indent=4, ensure_ascii=False, default=str)

    
