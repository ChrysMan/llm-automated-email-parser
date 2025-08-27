import json, os, sys
import networkx as nx
from typing import Optional, List, Dict
from utils.logging_config import LOGGER
from langchain.agents import AgentExecutor, create_structured_chat_agent
from tools.email_processor import split_and_extract_emails
from tools.build_graph import build_email_graph
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

class EmailAgent:
    def __init__(self):
        self.llm = ChatOllama(model="llama3.1", temperature=0.3)
        self.tools = [split_and_extract_emails, build_email_graph]
        self.graph = nx.DiGraph()
        self.email_data = []

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])

        self.agent = create_structured_chat_agent(self.llm, self.tools, self.prompt, Verbose=True)  # might add prompt
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, Verbose=True, handle_parsing_errors=True)

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
    
    def run(self, user_input: str, chat_history: Optional[List] = None) -> Dict:
        try: 
            # Initialize history if not provided
            chat_history = chat_history or []

            result = self.agent_executor.invoke({"input": user_input, "chat_history": chat_history})

            chat_history.extend([
                HumanMessage(content=user_input),
                AIMessage(content=str(result.get("output", "")))
            ])

            return {
                "output": result.get("output"),
                "used_tools": [tool.name for tool in self.agent_executor.tools_used],
                "chat_history": chat_history
            }
        except Exception as e:
            return (f"Agent failed: {e}")


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

    email_agent = EmailAgent()

    for filename in os.listdir(dir_path):
        if filename.endswith(".msg"):
            file_path = os.path.abspath(os.path.join(dir_path, filename))
            try:               
                result = email_agent.run(f"Split and extract the information from this email thread: {file_path}")
                email_data.extend(result["output"])
                result = email_agent.run(f"Build a graph from the same email thread : {graph}")
               # email_data.extend(result) 
            except Exception as e:
                LOGGER.error(f"Processing {filename} failed: {e}")
    #I want to save the results from split_and_extract_emails in a json file
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(email_data, file, indent=4, ensure_ascii=False, default=str)
    # visualize the graph

    


