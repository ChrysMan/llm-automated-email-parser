import os, asyncio, shutil
from dataclasses import dataclass
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from lightrag import LightRAG, QueryParam
from langchain_neo4j import Neo4jGraph
from lightrag_implementation.basic_operations import initialize_rag, index_data
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from dotenv import load_dotenv
import argparse

load_dotenv()

WORKING_DIR = "./lightrag_implementation/rag_storage"
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

model = OpenAIChatModel(
    os.getenv("LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8"),
    provider = OpenAIProvider(
        base_url=os.getenv("LLM_BINDING_HOST"), 
        api_key=os.getenv("LLM_BINDING_API_KEY")
    )    
)

@dataclass
class KGDeps:
    """Dependencies for the KG agent."""
    lightrag: LightRAG

# Create the Pydantic AI agent
agent = Agent(
    model,
    deps_type=KgDeps,
    system_prompt="""You are an Enterprise Email Intelligence Agent managing a LightRAG pipeline and Neo4j Knowledge Graph.

    You have access to tools: add_data, reinitialize_rag_storage, and close.
    If you need to use a tool, generate the tool call. Once the tool returns a result, summarize that result for the user.

    OPERATIONAL RULES:
    1. Data Integrity: Only use `add_data` when provided a directory path. Verify the existence of the directory before proceeding.
    2. Safety: `reinitialize_rag_storage` is destructive. Use ONLY if the user explicitly asks to "wipe," "reset," or "clear" the entire system.
    3. Finalization: You MUST call `close` when the user signals the end of a session (e.g., "exit", "stop", "bye") to ensure data is saved and connections are closed safely.
    
    TONE: Professional, secure."""
)

@agent.tool
async def reinitialize_rag_storage(ctx: RunContext[KGDeps]) -> str:
    """Deletes all data in the RAG storage and Neo4j Graph, then creates the clean LightRAG working directory and reinitializes LightRAG.
    
    Args:
        ctx (RunContext[KGDeps]): The run context containing dependencies.

    Returns:
        str: Confirmation message upon completion.
    """
    if ctx.deps.lightrag:
          await ctx.deps.lightrag.finalize_storages()
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

    working_dir = ctx.deps.lightrag.working_dir
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
        new_rag = await initialize_rag(working_dir=working_dir)
        ctx.deps.lightrag = new_rag
    except Exception as e:
        return f"Error reinitializing LightRAG: {e}"
    
    return "RAG storage has been cleared and LightRAG has been reinitialized."

@agent.tool
async def add_data(ctx: RunContext[KGDeps], dir_path: str) -> str:
    """Adds data from the given file path into the RAG system.
    
    Args:
        ctx (RunContext[KGDeps]): The run context containing dependencies.
        dir_path (str): The directory path containing data files to index.

    Returns:
        str: Confirmation message upon completion or Error message upon failure.
    """
    if not os.path.isdir(dir_path):
        return f"{dir_path} is not a valid directory."
    return await index_data(ctx.deps.lightrag, dir_path)

@agent.tool
async def close(ctx: RunContext[KGDeps]) -> str:
    """
    Safely finalize and close the LightRAG pipeline when instructed to "exit", "quit" or "stop"

    Args:
        ctx (RunContext[KGDeps]): The run context containing dependencies.

    Returns:
        str: Confirmation message upon successful closing
    """
    rag = ctx.deps.lightrag
    if rag is None:
        return "No active RAG instance found. Nothing to close." 
    try:
        # Ensure the method exists
        if hasattr(rag, "finalize_storages"):
            await rag.finalize_storages()
        else:
            return "Warning: LightRAG instance does not support finalize_storages()."

        return "Safely finalized the RAG pipeline. You may exit."
    
    except Exception as e:
        return f"Error while closing the RAG pipeline: {e}"
    
async def run_rag_agent(question: str) -> str:
    """Run the RAG agent to answer a question about Pydantic AI.
    
    Args:
        question: The question to answer.
        
    Returns:
        The agent's response.
    """
    # Create dependencies
    lightrag = await initialize_rag()
    deps = KGDeps(lightrag=lightrag)
    
    # Run the agent
    result = await agent.run(question, deps=deps)
    
    return result.output

def main():
    """Main function to parse arguments and run the kg agent."""
    parser = argparse.ArgumentParser(description="This script queries the PydanticAI agent with a user question.")
    parser.add_argument("--question", default="Add data to the graph from the directory /home/chryssida/DATA_TUC-KRITI/SEA IMPORT/234107.", help="The question to answer")

    args = parser.parse_args()

    # Run the agent
    response = asyncio.run(run_rag_agent(args.question))

    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    main()