import json, os, sys
import networkx as nx
import matplotlib.pyplot as plt
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

    
    try:
        for n, email in enumerate(raw_model_output):
            n+=1
            graph.add_node(n, email_node=email)
            if n > 1: 
                graph.add_edge(n-1, n)
    except Exception as e:
        LOGGER.error(f"Failed to create graph: {e}")


    return raw_model_output

def add_to_graph(graph: nx.DiGraph, email_data: List[str], filename: str) -> nx.DiGraph:
    """
    Add the email data to the graph.
    """
    LOGGER.info("Adding to Graph...")
    try:
        previous_key = None
        for n, email in enumerate(email_data):
            n += 1
            ''' Check for duplicate nodes '''
            for node_key, data in graph.nodes(data=True):
                if data['email_node'] == email:
                    previous_key = node_key
                    break
            ''' 
                Continue adding nodes after the duplicate node 
                If no duplicate node is found, add the node normally
            '''
            if previous_key:
                previous_key = int(previous_key.split("_")[0])
                key = f"{previous_key+1}_{filename}"
                graph.add_node(key, email_node=email)
                graph.add_edge(previous_key, key)
                previous_key += 1
            else: 
                key = f"{n}_{filename}"
                graph.add_node(key, email_node=email)
                if n > 1: 
                    graph.add_edge(f"{n-1}_{filename}", key)
        LOGGER.info(f"Added {len(email_data)} emails to the graph.")
    except Exception as e:
        LOGGER.error(f"Failed to add to graph: {e}")
    LOGGER.info("Graph updated.")
    return graph

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
            file_path = os.path.abspath(os.path.join(dir_path, filename))
            try:               
                result = split_emails(file_path)
                graph = add_to_graph(graph, result, filename)
                email_data.extend(result) 
            except Exception as e:
                LOGGER.error(f"Processing {filename} failed: {e}")
    
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(email_data, file, indent=4, ensure_ascii=False, default=str)
    
    nx.draw(graph, with_labels=True, node_color='skyblue', node_size=2000, font_size=16, arrows=True)
    plt.savefig("graph.png")

    
