import extract_msg, json, re, os, sys, ollama
import networkx as nx
from networkx.readwrite import json_graph
from langgraph.graph import StateGraph, START, END
from typing import List, Optional, TypedDict, Literal
from utils.logging_config import LOGGER
from utils.graph_utils import extract_msg_file, write_file, append_file
from chains.email_parser import EMAIL_PARSER_CHAIN
from chains.split_emails import SPLIT_EMAILS_CHAIN

class EmailProcessingState(TypedDict): 
    email_graph: nx.DiGraph             
    current_email: Optional[int]

class InputState(TypedDict):
    msg_content: str

''' Will use later
class OutputState:
    #xml
'''
            
def split_emails(state: InputState) -> EmailProcessingState:
    # Splits the emails from the .msg and creates a directed graph
    # with the chronological order of the emails   

    LOGGER.info("Splitting emails...")
    try:
        email_list = SPLIT_EMAILS_CHAIN.invoke({"text": state["msg_content"]})
        write_file("", "emailSplit")
        for email in email_list:
            append_file(email, "emailSplit")

        with open("emailInfo", "w") as file:
            pass
    except Exception as e:
        LOGGER.error(f"Failed to split emails: {e}")

    graph = nx.DiGraph()

    LOGGER.info("Creating Graph...")
    try:
        for n, email in enumerate(email_list):
            n+=1
            graph.add_node(n, email_node=email)
            if n > 1: 
                graph.add_edge(n-1, n)
    except Exception as e:
        LOGGER.error(f"Failed to create graph: {e}")

    return {"email_graph": graph, "current_email": 1}

def extract_email_info(state: EmailProcessingState) -> EmailProcessingState:
    # Use the email parser chain to extract fields from the emails
    LOGGER.info("Extract email fields...")

    graph = state["email_graph"]

    if state["current_email"]:
        try:
            # Extract fields
            if isinstance(graph.nodes[state["current_email"]].get("email_node"), str):
                graph.nodes[state["current_email"]]["email_node"] = EMAIL_PARSER_CHAIN.invoke({"email": graph.nodes[state["current_email"]].get("email_node")})

            # Find next email
            successors = list(graph.successors(state["current_email"]))
            next_email_id = successors[0] if successors else None
        except Exception as e:
            LOGGER.error(f"Failed to extract fields: {e}")

        return {"email_graph": graph, "current_email": next_email_id}

    return {"email_graph": graph, "current_email": state["current_email"]}

def decide_edge(state: EmailProcessingState) -> Literal["extract_email_info", END]:
    if state["current_email"] is None:
        return END
    else:
        return "extract_email_info"


def run_pipeline(file_path: str):
    
    raw_msg_content = extract_msg_file(file_path)
    #initial_state = InputState({"msg_content": raw_msg_content})

    workflow = StateGraph(EmailProcessingState, input=InputState, output=EmailProcessingState) #output=OutputState  

    workflow.add_node("split_emails", split_emails)
    workflow.add_node("extract_email_info", extract_email_info)

    workflow.add_edge(START, "split_emails")
    workflow.add_edge("split_emails", "extract_email_info")
    workflow.add_conditional_edges("extract_email_info", decide_edge)

    email_agent_graph = workflow.compile()

    #workflow.get_graph().print_mermaid()

    final_state = email_agent_graph.invoke(InputState({"msg_content": raw_msg_content})) 

    return final_state["email_graph"]


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

    emails = []

    for filename in os.listdir(dir_path):
        if filename.endswith(".msg"):
            file_path = os.path.join(dir_path, filename)
            try:
                #raw_msg_content = extract_msg_file(file_path)
                #email_data = email_agent_graph.invoke(raw_msg_content) run_pipeline
                email_data = run_pipeline(file_path) 
                graph_data = json_graph.node_link_data(email_data["email_graph"])
                for node in graph_data["nodes"]:
                    emails.append({
                        "id": node["id"],
                        "reply_to": node.get("reply_to"),
                        "email": node.get("email")
                    })
            except Exception as e:
                LOGGER.error(f"Processing {filename} failed: {e}")

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(emails, file, indent=4, ensure_ascii=False, default=str)

    


