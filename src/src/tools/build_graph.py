import networkx as nx
import matplotlib.pyplot as plt
from typing import List
from utils.logging_config import LOGGER
from pydantic import BaseModel, Field
from typing import List, Dict
from langchain.tools import tool

def build_email_graph(graph: nx.DiGraph, email_data: List[Dict], filename: str) -> nx.DiGraph:
    """
    Add emails to a graph based on their chronological order.

    Args:
        graph (nx.DiGraph): The base graph structure to augment.
        email_data (List[Dict]): List of processed emails from split_and_extract_emails.
        filename (str): Output filename
    Returns:
        nx.DiGraph: The updated graph with new email nodes and edges.
    """
    LOGGER.info("Adding to Graph...")
    try:
        filename_id = filename.split('-', 1)[1].split('.')[0]
        previous_key = None
        start_idx = 0 # was 1
        
        for idx, email in enumerate(email_data):
            for node_key, data in graph.nodes(data=True):
                if data['email_node'] == email:
                    previous_key = node_key
                    start_idx = idx

        for idx, email in enumerate(email_data[start_idx:], start=start_idx):
            ''' Adding new email thread without duplicates '''
            prev_id = int(previous_key.split('-')[0]) if previous_key else 0
            key = f"{prev_id+1}-{filename_id}"

            graph.add_node(key, email_node=email)

            if previous_key: #idx>1
                graph.add_edge(previous_key, key)

            previous_key = key
        LOGGER.info(f"Added {len(email_data)-start_idx} emails to the graph.")

    except Exception as e:
        LOGGER.error(f"Failed to add to graph: {e}")
        raise
    LOGGER.info("Graph update completed.")
    return graph