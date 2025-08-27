import json, os, sys
import networkx as nx
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage, ToolMessage
from typing import TypedDict, List, Annotated,Union
from utils.logging_config import LOGGER
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_structured_chat_agent
from chains.email_processor import split_and_extract_emails
from tools.build_graph import build_email_graph
from langchain_ollama import ChatOllama

class EmailAgent:
    def __init__(self, model):
        self.tools = [split_and_extract_emails, build_email_graph]
        self.model = model.bind_tools(self.tools)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])

        self.agent = create_structured_chat_agent(
            model = self.model,
            tools = self.tools,
            prompt = self.prompt)
        
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, Verbose=True, handle_parsing_errors=True)
        
        self.graph = nx.DiGraph()  # Single source of truth
        
        # Create reusable workflow
        self.workflow = self._create_workflow()
    
    def _get_system_prompt(self) -> str:
        return """You're an advanced email processing system. Your available tools are:
        
        1. split_and_extract_emails: Use for:
        - Raw email thread splitting
        - Metadata extraction
        - When you need individual email components
        
        2. build_email_graph: Use for:
        - Visualizing sender-recipient relationships
        - Timeline analysis
        - When asked to "show connections" or "visualize"
        
        Think step-by-step before selecting tools."""


    def _create_workflow(self):
        class ProcessState(TypedDict):
            messages: Annotated[List[AnyMessage], operator.add]   
            current_graph: nx.DiGraph  # Temporary working copy

        builder = StateGraph(ProcessState)
        
        def agent_node(state: ProcessState):
            result = self.agent_executor.invoke({
                "input": state["messages"][-1].content,
                "chat_history": state["messages"],
                "current_graph": state["current_graph"]
            })

            new_message = AIMessage(content=result["output"])

            return {
                "messages": state["messages"] + [new_message],
                "current_graph": result.get("current_graph", nx.DiGraph())
            }

        def tool_node(state: ProcessState):
            last_msg = state["messages"][-1]
            tool_name = last_msg.additional_kwargs.get("tool_call", {}).get("name")
            
            
            # Process file-scoped data
            if tool_name == "split_and_extract_emails":
                file_path = last_msg.content
                email_data = split_and_extract_emails(file_path)
                
                output_path = os.path.join(
                    os.path.dirname(file_path), 
                    f"{os.path.basename(file_path).split('.')[0]}.json"
                )

                # Save email data to JSON file
                with open(output_path, "w", encoding="utf-8") as file:
                    json.dump(email_data, file, indent=4, ensure_ascii=False, default=str)
                
                # Update working graph copy
                return {
                    "current_graph": build_email_graph(
                        state["current_graph"], 
                        email_data,
                        file_path.split('/')[-1]  # Extract filename
                    )
                }
            
            return state

        builder.add_node("agent", agent_node)
        builder.add_node("tool", tool_node)
        builder.add_edge("agent", "tool")
        builder.add_edge("tool", "agent")
        builder.set_entry_point("agent")
        
        return builder.compile()

    def process_file(self, file_path: str):
        """Process a single file and update persistent graph"""
        # Create temporary working state
        result = self.workflow.invoke({
            "messages": [HumanMessage(content=f"Process {file_path}")],
            "current_graph": self.graph.copy()  # Work on copy
        })
        
        # Commit final graph state
        self.graph = result["current_graph"]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        LOGGER.error("Usage: python emailParsing.py <dir_path>")
        sys.exit(1)
    
    dir_path = sys.argv[1]

    if not os.path.isdir(dir_path):
        LOGGER.error(f"{dir_path} is not a valid directory.")
        sys.exit(1)

    output_path = os.path.join(dir_path, f"{os.path.basename(dir_path)}.json")

    model = ChatOllama(model="llama3.1", temperature=0.3)
    email_agent = EmailAgent(model)

    for filename in os.listdir(dir_path):
        if filename.endswith(".msg"):
            file_path = os.path.abspath(os.path.join(dir_path, filename))
            try:               
                result = email_agent.process_file(file_path)
    
               # email_data.extend(result) 
            except Exception as e:
                LOGGER.error(f"Processing {filename} failed: {e}")
