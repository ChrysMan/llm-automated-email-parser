import os, asyncio, json
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from langsmith import traceable
from dotenv import load_dotenv
from lightrag.tools.clean_llm_query_cache import CleanupTool

from ..core.pipeline import initialize_rag, index_data
from ..agents.dependencies import AgentDeps
from ..core.llm import agent_llm
from utils.logging import LOGGER
from utils.file_io import find_file, read_json_file

load_dotenv()

# if os.getenv("LANGSMITH_API_KEY"):
#     os.environ["LANGCHAIN_TRACING_V2"] = "true"
#     os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
#     os.environ["LANGSMITH_PROJECT"] = "kg_agent"
# else:
#     LOGGER.warning("Langsmith API key not found. Tracing will be disabled.")

WORKING_DIR = "lightrag_impl/core/rag_storage"

kg_agent = Agent(
    agent_llm,
    deps_type=AgentDeps,
    retries=3,
    model_settings={'parallel_tool_calls': False},
    system_prompt="""You are an Enterprise Email Intelligence Agent managing a LightRAG pipeline and Neo4j Knowledge Graph for secure email data processing.

TOOLS:
1. add_data: Index email data from directories into the knowledge graph.
2. delete_rag_storage: Remove ALL indexed data and graph content (destructive).
3. clear_cache: Clear LLM response caches while preserving indexed data.
4. close: Safely shutdown the RAG pipeline and release resources.

CRITICAL GUIDELINES:
1. Tool Execution: Only call tools when explicitly requested. Never act proactively.
2. Data Operations: Use add_data for indexing requests from specified directories.
3. Destructive Actions: 
   - delete_rag_storage: Only for explicit "delete entire system/graph" requests.
   - clear_cache: Only for "clear cache/history/responses" requests.
4. System Control: Use close only for explicit shutdown/finalize requests.
5. Sequential Processing: Execute tools one at a time with clear feedback.
6. Error Management: Report failures clearly and suggest alternatives.

COMMUNICATION: Maintain professional, secure, and detailed responses based on tool outputs."""
)


@traceable
@kg_agent.tool 
async def delete_rag_storage(ctx: RunContext[AgentDeps]) -> str:
    """Deletes all data and response cache in the RAG storage and Neo4j Graph.
    Use this tool ONLY when the user explicitly requests to "delete", "wipe," "reset," or "clear" the entire system - it is a destructive operation.
    
    Args:
        ctx (RunContext[AgentDeps]): The run context containing dependencies.

    Returns:
        str: Confirmation message upon completion.
    """

    target_file = "kv_store_full_docs.json"
    target_file_doubleCheck = "kv_store_entity_chunks.json"

    file_path = find_file(target_file, WORKING_DIR)
    file_path_doubleCheck = find_file(target_file_doubleCheck, WORKING_DIR)

    if not file_path:
        LOGGER.error(f"Could not find {target_file} in {WORKING_DIR} or its subdirectories.")
        return f"Could not find {target_file} in {WORKING_DIR} or its subdirectories."
    
    if ctx.deps.lightrag:
        try:
            dicts = read_json_file(file_path)
        
            for doc_id, _ in dicts.items():
                await ctx.deps.lightrag.adelete_by_doc_id(doc_id, True)

            dicts_doubleCheck = read_json_file(file_path_doubleCheck)
            
            for entity, _ in dicts_doubleCheck.items():
                await ctx.deps.lightrag.adelete_by_entity(entity)

        except Exception as e:
            LOGGER.error(f"Error during deletion of documents: {e}")
            return f"Error during deletion of documents: {e}"
    
    return "RAG storage and cache has been cleared."

@traceable
@kg_agent.tool
async def clear_cache(ctx: RunContext[AgentDeps]) -> str:
    """Clear all caches responses from the LLM response cache storage.
    Args:
        ctx (RunContext[AgentDeps]): The run context containing dependencies.

    Returns:
        str: Confirmation message upon completion or Error message upon failure.

    """
    try:
        await ctx.deps.lightrag.aclear_cache()
    except Exception as e:
        LOGGER.error(f"Error while trying to clear response cache: {e}")
        return f"Error while trying to clear LLM response cache storage: {e}"
    return "Succesfully cleared LLM response cache storage."

@traceable
@kg_agent.tool
async def add_data(ctx: RunContext[AgentDeps], dir_path: str | None = None) -> str:
    """Adds data from the given directory into the RAG system.
    
    Args:
        ctx (RunContext[AgentDeps]): The run context containing dependencies.
        dir_path (str): The directory path containing data files to index.

    Returns:
        str: Confirmation message upon completion or Error message upon failure.
    """
    if dir_path is None or dir_path.strip() == "":
        if ctx.deps.dir_path is not None:
            dir_path = ctx.deps.dir_path
        else:
            return "Please provide a valid directory path for adding data to the graph."
    else:
        ctx.deps.dir_path = dir_path

    if not os.path.isdir(dir_path):
        return f"{dir_path} is not a valid directory. Provide a correct path containing preprocessed 'unique.json' files."
    return await index_data(ctx.deps.lightrag, dir_path)

@traceable
@kg_agent.tool
async def close(ctx: RunContext[AgentDeps]) -> str:
    """
    Safely finalize and close the LightRAG pipeline when instructed to "exit", "quit", "close" or "stop".
    Do not call this tool unless explicitly requested by the user.

    Args:
        ctx (RunContext[AgentDeps]): The run context containing dependencies.

    Returns:
        str: Confirmation message upon successful closing
    """
    rag = ctx.deps.lightrag
    if rag is None:
        return "No active RAG instance found. Nothing to close." 
    try:
        if hasattr(rag, "finalize_storages"):
            await rag.finalize_storages()
        else:
            return "Warning: LightRAG instance does not support finalize_storages()."

        return "Safely finalized the RAG pipeline. You may exit."
    
    except Exception as e:
        return f"Error while closing the RAG pipeline: {e}"

async def run_interactive_loop():
    """Starts a continuous chat session with the KG Agent."""
    lightrag = await initialize_rag()
    deps = AgentDeps(lightrag=lightrag)
    
    message_history = []
    
    print("\n--- KG Agent Interactive Session (Type 'exit' to quit) ---")

    while True:
        user_input = input("\nUser: ").strip()
        print("\n")
        
        if user_input.lower() in ["exit", "quit", "stop", "bye"]:
            await kg_agent.run("Close the pipeline and exit.", deps=deps, message_history=message_history)
            print("Goodbye!")
            break

        if not user_input:
            continue

        try:
            result = await kg_agent.run(
                user_input, 
                deps=deps, 
                message_history=message_history
            )
            
            message_history = result.all_messages()
            print(f"\nAgent: {result.output}")

        except Exception as e:
            print(f"\n[SYSTEM ERROR]: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(run_interactive_loop())
    except KeyboardInterrupt:
        print("\nSession interrupted.")