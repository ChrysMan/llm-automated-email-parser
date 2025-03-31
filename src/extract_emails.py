import extract_msg, json, re, os, sys, ollama
import networkx as nx
from networkx.readwrite import json_graph
from typing import List, Optional, TypedDict, Literal
from utils.logging_config import LOGGER
from utils.graph_utils import extract_msg_file, write_file, append_file, split_email_chain
from chains.email_parser import EMAIL_PARSER_CHAIN, EmailInfo
from chains.split_emails import SPLIT_EMAILS_CHAIN
from langchain_core.output_parsers import CommaSeparatedListOutputParser, ListOutputParser

def split_emails(msg: str):
    LOGGER.info("Splitting emails...")
    try:
        raw_model_output = SPLIT_EMAILS_CHAIN.invoke({"emails": msg})
        
        LOGGER.info("Splitted emails...")
        
        #print(email_list)
        reversed_list = raw_model_output[::-1]
    
    except Exception as e:
        LOGGER.error(f"Failed to split emails: {e}")

    graph = nx.DiGraph()

    LOGGER.info("Creating Graph...")
    try:
        for n, email in enumerate(reversed_list):
            n+=1
            graph.add_node(n, email_node=email)
            if n > 1: 
                graph.add_edge(n-1, n)
    except Exception as e:
        LOGGER.error(f"Failed to create graph: {e}")


    return reversed_list

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

    write_file("", output_path)

    for filename in os.listdir(dir_path):
        if filename.endswith(".msg"):
            file_path = os.path.join(dir_path, filename)
            try:
                raw_msg_content = extract_msg_file(file_path)
                clean_msg_content = re.sub(r"^\s+|\s{2,}", "\n", raw_msg_content)
                # choose between raw_msf_content or clean_msg_content
                email_data = split_emails(raw_msg_content) 
                with open(output_path, "a", encoding="utf-8") as file:
                    json.dump(email_data, file, indent=4, ensure_ascii=False, default=str)
            except Exception as e:
                LOGGER.error(f"Processing {filename} failed: {e}")

    
