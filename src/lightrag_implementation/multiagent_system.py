import os, json, re, shutil, threading
from time import time
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage,AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool, tool
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from lightrag import LightRAG, QueryParam
from typing import List, TypedDict
from basic_operations import initialize_rag, index_data
from src.email_preprocessing_vllm_serve import LLMPredictor
from src.lightrag_implementation.basic_operations import initialize_rag
from src.deduplicate_cleaned_emails import deduplicate_emails
from src.utils.prompts import cleaning_prompt,formatter_and_translator_prompt
from src.utils.graph_utils import extract_msg_file, clean_data, split_email_thread
from src.utils.logging_config import LOGGER
from dotenv import load_dotenv

load_dotenv()

import operator

# Define tools
# class Basic_toolTool(BaseTool):
#     name = "basic_tool"
#     description = "Tool for basic_tool operations"$\rightarrow$
    
#     def _run(self, query: str) -> str:
#         # Implement actual functionality here
#         return f"Result from basic_tool tool: {query}"
    
#     async def _arun(self, query: str) -> str:
#         # Implement actual functionality here
#         return f"Result from basic_tool tool: {query}"

# tools = [
#     Basic_toolTool(),
# ]

class RefineQueryInput(BaseModel):
    """Input for the rephrase_and_refine_query tool."""
    user_query: str = Field(description="The complex, conversational query provided by the user.")

class RefinedQueries(BaseModel):
    """The structured output containing the optimal search queries."""
    refined_queries: List[str] = Field(
        description="A list of 1 to 3 highly specific, optimal search queries suitable for running against the vector store."
    )
    reasoning: str = Field(
        description="A brief explanation of the thought process behind the refinement."
    )

@tool
def execute_full_preprocessing(dir_path: str)-> str:
    """Preprocess emails in the given directory and return a list of cleaned and unique email texts."""
    tic = time()

    if not os.path.isdir(dir_path):
        return f"{dir_path} is not a valid directory."
    
    folder_name = os.path.basename(os.path.normpath(dir_path))
    
    output_path = os.path.join(dir_path, f"{folder_name}_unique.json")

    predictor = LLMPredictor()

    all_emails_to_process = []
    for filename in os.listdir(dir_path):
        if filename.endswith(".msg"):
            file_path = os.path.join(dir_path, filename)

            try:
                raw_msg_content = extract_msg_file(file_path)
                cleaned_msg_content = clean_data(raw_msg_content)
                all_emails_to_process.extend(split_email_thread(cleaned_msg_content))

            except Exception as e:
                LOGGER.error(f"Failed to extract or clean email from {filename}: {e}")
                continue

    # Prepare all prompts outside the file loop
    formatting_prompts = [formatter_and_translator_prompt.format(email=e) for e in all_emails_to_process]
    results = predictor(formatting_prompts)

    str_results = [str(r) for r in results]

    cleaning_prompts = [cleaning_prompt.format(email=e) for e in str_results]
    results = predictor(cleaning_prompts)

    # ---------------Deduplicate results---------------
    unique_emails = deduplicate_emails(results)
    emails_json = []
    for text in unique_emails:
            # Split at the first "body:" (case-insensitive, multi-line safe)
            parts = re.split(r'(?mi)^\s*body\s*:\s*', text, maxsplit=1)
            headers_part = parts[0]
            body_part = parts[1] if len(parts) == 2 else ""

            email_dict = {}
            for line in headers_part.splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)  # split only on first colon
                    email_dict[key.strip().lower()] = val.strip()

            email_dict["body"] = body_part.rstrip()
            emails_json.append(email_dict)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(emails_json, file, indent=4, ensure_ascii=False, default=str)

    return f"The preprocessing has completed in {time() - tic} seconds and the output is saved at {output_path}."

@tool
async def reinitialize_kg(lightrag: LightRAG) -> str:
    """Deletes all data in the RAG storage and Neo4j Graph. Reinitializes LightRAG.
    
    Args:
        lightrag (LightRAG): The LightRAG instance to query.

    Returns:
        str: Confirmation message upon completion.
    """  
    from langchain_neo4j import Neo4jGraph

    if lightrag:
        lightrag.finalize_storages()
        try:
            graph = Neo4jGraph(
                url=os.getenv('NEO4J_URI'),
                username=os.getenv('NEO4J_USERNAME'),
                password=os.getenv('NEO4J_PASSWORD')
            )
            graph.query("MATCH (n) DETACH DELETE n")
            graph.close()
        except Exception as e:
            return f"Error connecting to Neo4j: {e}"

        working_dir = lightrag.working_dir
        if os.path.exists(working_dir):
            try:
                shutil.rmtree(working_dir)
            except Exception as e:
                return f"Error clearing RAG storage contents: {e}"
            
        try:
            os.makedirs(working_dir)
        except Exception as e:
            return f"Error recreating RAG storage directory: {e}"
        
    try:
        new_rag = initialize_rag(working_dir=working_dir)
        lightrag = new_rag
    except Exception as e:
        return f"Error reinitializing LightRAG: {e}"
    
    return "RAG storage cleared and LightRAG reinitialized."

@tool
async def add_to_kg(lightrag: LightRAG, dir_path: str) -> str:
    """Adds data from the given file path into the RAG system.
    
    Args:
        lightrag (LightRAG): The LightRAG instance to query.
        dir_path (str): The directory path containing data files to index.

    Returns:
        str: Confirmation message upon completion or Error message upon failure.
    """
    return await index_data(lightrag, dir_path)

@tool
async def close_pipeline(lightrag: LightRAG) -> str:
    """
    Safely finalize and close the LightRAG pipeline when instructed to "exit", "quit" or "stop"

    Args:
        lightrag (LightRAG): The LightRAG instance to query

    Returns:
        str: Confirmation message upon successful closing
    """
    if lightrag is None:
        return "No active RAG instance found. Nothing to close." 
    try:
        # Ensure the method exists
        if hasattr(lightrag, "finalize_storages"):
            await lightrag.finalize_storages()
        else:
            return "Warning: LightRAG instance does not support finalize_storages()."

        return "Safely finalized the RAG pipeline. You may exit."
    
    except Exception as e:
        return f"Error while closing the RAG pipeline: {e}"

@tool
async def retrieve(lightrag: LightRAG, question: str) -> str:
    """
    Retrieve relevant documents from the RAG system based on the question.
    Args:
        lightrag (LightRAG): The LightRAG instance to query.
        question (str): The question to retrieve documents for.

    Returns:
        str: The retrieved documents as a string.
    """
    return await lightrag.aquery(
        query=question,
        param=QueryParam(mode="mix", enable_rerank=True, include_references=True)
    )

class RefineQueryTool(BaseTool):
    name = "rephrase_and_refine_query"
    description = (
        "Analyzes a conversational user query and generates 1-3 optimized, specific, "
        "and concise search queries that will maximize retrieval accuracy from the email Knowledge Graph. "
        "Use this ONLY when the user's query is complex or ambiguous."
    )
    args_schema: RefineQueryInput = RefineQueryInput
    
    llm = ChatOpenAI(temperature=0, model="Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8", base_url="http://localhost:8001/v1", api_key="EMPTY")
    
    def _create_refinement_chain(self):
        """Creates the internal LLM chain for query refinement."""
        template = """
        You are an expert Query Refinement Agent. Your task is to analyze the user's query
        about email data and generate 1 to 3 highly specific, concise queries that will maximize retrieval 
        accuracy from a vector database.

        The queries should focus on key entities, dates, and topics found in the user's input.
        
        CONVERSATIONAL QUERY: {query}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        return prompt | self.llm.with_structured_output(RefinedQueries)

    # --- Execution Method (Synchronous) ---
    def _run(self, user_query: str) -> RefinedQueries:
        """The synchronous execution of the tool."""
        
        try:
            refinement_chain = self._create_refinement_chain()
            
            result: RefinedQueries = refinement_chain.invoke({"query": user_query})
            
            return result
        
        except Exception as e:
            return RefinedQueries(
                refined_queries=[],
                reasoning=f"Technical failure during refinement: {str(e)}"
            )

    # --- Execution Method (Asynchronous ---
    async def _arun(self, user_query: str):
        """Placeholder for asynchronous execution."""
        try:
            refinement_chain = self._create_refinement_chain()
            
            result: RefinedQueries = refinement_chain.invoke({"query": user_query})
            
            return result
        except Exception as e:
            return RefinedQueries(
                refined_queries=[],
                reasoning=f"Technical failure during async refinement: {str(e)}"
            )
    
refine_tool = RefineQueryTool()
    


# Define state
class AgentState(TypedDict):
    messages: List[BaseMessage]
    next: str
    lightrag: LightRAG
    job_id: Optional[str]
    job_status: Literal["IDLE", "RUNNING", "COMPLETE", "FAILED"]
    clean_data_path: Optional[str]


# Agent: default_assistant
def default_assistant_agent(state: AgentState) -> AgentState:
    """Agent that handles General Assistant."""
    # Create LLM
    llm = ChatOpenAI(model="gpt-4.1-mini")
    # Get the most recent message
    messages = state['messages']
    response = llm.invoke(messages)
    # Add the response to the messages
    return {
        "messages": messages + [response],
        "next": state.get("next", "")
    }

# Define routing logic
def router(state: AgentState) -> str:
    """Route to the next node."""
    return state.get("next", "END")

# Define the graph
workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("process_input", default_assistant_agent)

# Add conditional edges
workflow.add_edge("process_input", END)

# Set entry point
workflow.set_entry_point("process_input")

# Compile the graph
app = workflow.compile()

# Run the graph
def run_agent(query: str) -> List[BaseMessage]:
    """Run the agent on a query."""
    result = app.invoke({
        "messages": [HumanMessage(content=query)],
        "next": ""
    })
    return result["messages"]

# Example usage
if __name__ == "__main__":
    result = run_agent("Your query here")
    for message in result:
        print(f"{message.type}: {message.content}")