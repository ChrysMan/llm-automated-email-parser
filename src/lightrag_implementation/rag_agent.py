import os, sys, json, asyncio, shutil
from dataclasses import dataclass
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from lightrag import LightRAG, QueryParam
from langchain_neo4j import Neo4jGraph
from basic_operations import initialize_rag, index_data
from pydantic_ai.models.huggingface import HuggingFaceModel
from dotenv import load_dotenv
import argparse

load_dotenv()

WORKING_DIR = "./rag_storage"
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

#moonshotai/Kimi-K2-Thinking
model = HuggingFaceModel("moonshotai/Kimi-K2-Thinking")
@dataclass
class RAGDeps:
    """Dependencies for the RAG agent."""
    lightrag: LightRAG

# Create the Pydantic AI agent
agent = Agent(
    model, # fix
    deps_type=RAGDeps,
    system_prompt="""You are a knowledgeable assistant that uses a Retrieval-Augmented Generation (RAG) system to provide accurate and concise information based on retrieved documents. 
    Use the reinitialize_rag_storage to delete the graph and reinitialize rag. Use the add_data tool to add data into the graph. Use the retrieve tool when you want to retrieve information
    from the graph. Use the close tool to finalize and close the lightRAG pipeline. If the graph doesn't contain the answer, clearly state that the information isn't available in the graph.
    """
)

@agent.tool
async def reinitialize_rag_storage(context: RunContext[RAGDeps]) -> str:
    """Deletes all data in the RAG storage and Neo4j Graph. Reinitializes LightRAG.
    
    Args:
        context (RunContext[RAGDeps]): The run context containing dependencies.

    Returns:
        str: Confirmation message upon completion.
    """
    if context.deps.lightrag:
          await context.deps.lightrag.finalize_storages()
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

    working_dir = context.deps.lightrag.working_dir
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
        context.deps.lightrag = new_rag
    except Exception as e:
        return f"Error reinitializing LightRAG: {e}"
    
    return "RAG storage cleared and LightRAG reinitialized."

@agent.tool
async def add_data(context: RunContext[RAGDeps], dir_path: str) -> str:
    """Adds data from the given file path into the RAG system.
    
    Args:
        context (RunContext[RAGDeps]): The run context containing dependencies.
        dir_path (str): The directory path containing data files to index.

    Returns:
        str: Confirmation message upon completion or Error message upon failure.
    """
    await index_data(context.deps.lightrag, dir_path)
    return "Data indexing completed."

@agent.tool
async def retrieve(context: RunContext[RAGDeps], question: str) -> str:
    """
    Retrieve relevant documents from the RAG system based on the question.

    Args:
        context (RunContext[RAGDeps]): The run context containing dependencies.
        question (str): The user's question to retrieve information for.
    
    Returns:
        str: The retrieved information from the RAG system.
    """
    return await context.deps.lightrag.aquery(
        query=question,
        param=QueryParam(mode="mix", enable_rerank=True, include_references=True)
    )

@agent.tool
async def close(context: RunContext[RAGDeps]) -> str:
    """
    Safely finalize and close the LightRAG pipeline when instructed to "exit", "quit" or "stop"

    Args:
        context (RunContext[RAGDeps]): The run context containing dependencies.

    Returns:
        str: Confirmation message upon successful closing
    """
    rag = context.deps.lightrag
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
    deps = RAGDeps(lightrag=lightrag)
    
    # Run the agent
    result = await agent.run(question, deps=deps)
    
    return result.data

def main():
    """Main function to parse arguments and run the RAG agent."""
    parser = argparse.ArgumentParser(description="Who is Sofia Stafylaraki?")
    parser.add_argument("--question", help="The question to answer about Pydantic AI")

    args = parser.parse_args()

    # Run the agent
    response = asyncio.run(run_rag_agent(args.question))

    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    main()