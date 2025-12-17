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

class PreprocessingInput(BaseModel):
    dir_path: str = Field(description="The directory path containing the .msg email files to process.")

class ExecutePreprocessingTool(BaseTool):
    name = "execute_full_preprocessing"
    description = (
        "Executes the full pipeline to extract, clean, refine with LLM, and deduplicate emails "
        "in the specified directory. Returns the path to the resulting clean JSON file."
    )
    args_schema = PreprocessingInput

    def _run(self, dir_path: str) -> str:
        tic = time()
        if not os.path.isdir(dir_path):
            return {"status": "FAILED", "reason": f"{dir_path} is not a valid directory."}
        
        folder_name = os.path.basename(os.path.normpath(dir_path))
    
        output_path = os.path.join(dir_path, f"{folder_name}_unique.json")

        predictor = LLMPredictor()

        all_emails_to_process = []

        try:
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

        except Exception as e:
            return f"Error during preprocessing: {e}"

class ReinitializeKGInput(BaseModel):
    working_dir: str = Field(description="The working directory path for RAG storages.")

class ReinitializeKGTool(BaseTool):
    name = "reinitialize_kg"
    description = (
        "Deletes all data in the RAG storage and Neo4j Graph, then craetes the clean LightRAG working directory. "
        "Requires the RAG working directory path."
    )
    args_schema = ReinitializeKGInput

    def _run(self, working_dir: str):        
        from langchain_neo4j import Neo4jGraph
        
        # 1. Neo4j Cleanup
        graph = Neo4jGraph(
            url=os.getenv('NEO4J_URI'),
            username=os.getenv('NEO4J_USERNAME'),
            password=os.getenv('NEO4J_PASSWORD')
        )
        graph.query("MATCH (n) DETACH DELETE n") 
        graph.close() 

        # 2. Filesystem Cleanup
        if os.path.exists(working_dir):
            shutil.rmtree(working_dir)
        os.makedirs(working_dir) 
        
        # 3. Simulate LightRAG creation and return a confirmation object 
        # The Agent Node will then update the state with the actual LightRAG instance.
        return "RAG storage cleared and environment reinitialized."

    async def _arun(self, working_dir: str) -> str:
        return await super()._arun(working_dir)
    

class AddToKGInput(BaseModel):
    lightrag_instance_id: str = Field(description="The unique identifier/key for the active LightRAG instance required for indexing.")
    dir_path: str = Field(description="The directory path containing the clean data files (e.g., .json) to index.")

class AddToKGTool(BaseTool):
    name = "add_to_kg"
    description = (
        "Adds the pre-processed data from the specified directory path into the active RAG system index. "
        "This is a high-latency I/O operation."
    )
    args_schema = AddToKGInput

    # We omit the _arun method for the synchronous thesis requirement.

    def _run(self, lightrag_instance_id: str, dir_path: str) -> str:
        """Synchronous execution of data indexing (The user waits)."""
        
        # NOTE ON THESIS IMPLEMENTATION: 
        # The Agent Node must retrieve the actual LightRAG object from the AgentState 
        # using the 'lightrag_instance_id' and pass it here, or the tool accesses 
        # a global dictionary where the instance is stored (less clean). 
        # For simplicity, assume the Agent Node passes the actual object, and we just 
        # use the ID argument here for demonstration.
        
        try:
            # 1. Simulate finding the LightRAG instance
            # lightrag_instance = lookup_lightrag_instance(lightrag_instance_id) 
            
            # 2. Execute the synchronous indexing function
            # index_data(lightrag_instance, dir_path) 
            
            # 3. Simulate work time (This is the blocking part where the user waits)
            # You would replace the sleep with the actual index_data call:
            import time
            time.sleep(5) 
            
            return f"Data from {dir_path} indexed successfully by instance {lightrag_instance_id}."
        
        except FileNotFoundError:
            return f"Error: Clean data not found at {dir_path}. Did preprocessing complete?"
        except Exception as e:
            return f"Error during data indexing by {lightrag_instance_id}: {e}"
            
# Input Schema
class ClosePipelineInput(BaseModel):
    lightrag_instance_id: str = Field(description="The unique identifier/key for the active LightRAG instance to be safely closed.")


class ClosePipelineTool(BaseTool):
    name = "close_pipeline"
    description = (
        "Safely finalizes and closes the active LightRAG pipeline's storages (e.g., flushes buffers, closes DB connections). "
        "This should be the final action before application exit or a major reinitialization."
    )
    args_schema = ClosePipelineInput

    # We omit the _arun method for the synchronous thesis requirement.

    def _run(self, lightrag_instance_id: str) -> str:
        """Synchronous execution of pipeline finalization."""
        
        try:
            # 1. Simulate finding the LightRAG instance
            # lightrag_instance = lookup_lightrag_instance(lightrag_instance_id) 
            
            if lightrag_instance_id is None or lightrag_instance_id == "NONE":
                 return "No active RAG instance found. Nothing to close." 

            # 2. Execute the synchronous finalization function
            # lightrag_instance.finalize_storages()
            
            return "Safely finalized the RAG pipeline. You may exit."
        
        except Exception as e:
            # This tool should handle resource errors gracefully
            return f"Error while closing the RAG pipeline: {e}"