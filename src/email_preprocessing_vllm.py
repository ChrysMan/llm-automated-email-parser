import json, os, sys
import networkx as nx
import matplotlib.pyplot as plt
from typing import List
from utils.logging_config import LOGGER
from concurrent.futures import ProcessPoolExecutor
from utils.graph_utils import write_file, extract_msg_file, append_file, chunk_emails, clean_data, split_email_thread
from chains.split_emails import SPLIT_EMAILS_CHAIN 

os.environ["OLLAMA_SCHED_SPREAD"] = "1"

def process_chunk_wrapper(args):
    """Wrapper function to unpack arguments for process_chunk."""
    return process_chunk(*args)

def process_chunk(chunk: List[str], gpu_id: int) -> List[str]:
    """Process a chunk of emails on a specific GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # Assign specific GPU
    LOGGER.info(f"Processing on GPU {gpu_id}")

   
    chunk = "\n*** \n".join(chunk)
    #append_file(chunk, "emailSplit.txt")
    return SPLIT_EMAILS_CHAIN.invoke({"emails": chunk})
   
def split_emails(file_path: str, num_gpus: int = 4) -> List[str]:
    raw_msg_content = extract_msg_file(file_path)
    cleaned_msg_content = clean_data(raw_msg_content)
    append_file(cleaned_msg_content, "clean.txt")
    LOGGER.info("Splitting emails...")
    try:
        raw_model_output = []
        splitted_emails = split_email_thread(cleaned_msg_content)
        reversed_list = splitted_emails[::-1]
        chunks = list(chunk_emails(reversed_list, chunk_size=6))

        # Assign chunks to GPUs in a round-robin fashion
        gpu_chunks = [(chunk, gpu_id % num_gpus) for gpu_id, chunk in enumerate(chunks)]
        
         # Use multiprocessing to process chunks in parallel
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            results = list(executor.map(process_chunk, *zip(*gpu_chunks)))

        for result in results:
            if result:
                raw_model_output.extend(result)
        LOGGER.info("Splitted emails...")        
    
    except Exception as e:
        LOGGER.error(f"Failed to split emails: {e}")

    return raw_model_output

def add_to_graph(graph: nx.DiGraph, email_data: List[str], filename: str) -> nx.DiGraph:
    """
    Add the email data to the graph.
    """
    LOGGER.info("Adding to Graph...")
    try:
        filename = filename.split('-')[1].split('.')[0]
        previous_key = None
        start = 1
        
        for n, email in enumerate(email_data):
            for node_key, data in graph.nodes(data=True):
                if data['email_node'] == email:
                    previous_key = node_key
                    start = n

        for n, email in enumerate(email_data[start:], start=start):
            if previous_key is None: 
                ''' Adding new email thread without duplicates '''
                pr_key = 0
            else: 
                pr_key = int(previous_key.split("_")[0])
            key = f"{pr_key+1}_{filename}"
            graph.add_node(key, email_node=email)
            if n>1:
                graph.add_edge(previous_key, key)
            previous_key = key
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
            file_path = os.path.abspath(os.path.join(dir_path, filename))
            try:               
                result = split_emails(file_path)
                graph = add_to_graph(graph, result, filename)
                email_data.extend(result) 
            except Exception as e:
                LOGGER.error(f"Processing {filename} failed: {e}")
    
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(email_data, file, indent=4, ensure_ascii=False, default=str)
    

    plt.figure(figsize=(12, 10))
    pos = nx.kamada_kawai_layout(graph)
    node_options = {"node_color": "black", "node_size": 10}
    edge_options = {"width": 1, "alpha": 0.5, "edge_color": "black", "arrowsize": 5, "connectionstyle": 'arc3,rad=0.2'}
    label_options = {"font_size": 5, "font_color": "blue", "verticalalignment": "top", "horizontalalignment": "right"}
    nx.draw_networkx_nodes(graph, pos, **node_options)
    nx.draw_networkx_edges(graph, pos, **edge_options)
    nx.draw_networkx_labels(graph, pos, **label_options)
    plt.savefig("graph.png")

    
