import os, asyncio, json
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from langsmith import traceable
from dotenv import load_dotenv
from lightrag.tools.clean_llm_query_cache import CleanupTool

from lightrag_implementation.basic_operations import initialize_rag, index_data
from lightrag_implementation.agents.agent_deps import AgentDeps
from lightrag_implementation.llm import agent_llm
from utils.logging_config import LOGGER
from utils.graph_utils import find_file, read_json_file 

load_dotenv()

langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
if langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
    os.environ["LANGSMITH_PROJECT"] = "kg_agent"
else:
    LOGGER.warning("Langsmith API key not found. Tracing will be disabled.")

WORKING_DIR = "./lightrag_implementation/rag_storage"
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

kg_agent = Agent(
    agent_llm,
    deps_type=AgentDeps,
    retries=3,
    model_settings={'parallel_tool_calls': False},
    system_prompt="""You are an Enterprise Email Intelligence Agent managing a LightRAG pipeline and Neo4j Knowledge Graph.

    You have access to tools: add_data, delete_rag_storage, and close.
    Reason over which tool to call based on the user's request and call them appropriately. 
    You can call multiple tools in a single turn sequencially until the question is answered. Provide clear and detailed responses based on the outputs of the tools you invoke.

    OPERATIONAL RULES:
    1. Use each tool only when asked to do so by the user.
    2. Safety: `delete_rag_storage` is destructive. Use it ONLY when the user explicitly requests to "delete", "wipe," "reset," or "clear" the entire system.
    3. Finalization: You ONLY call `close` when the user requests to close or finalize the system to ensure data is saved and connections are closed safely.

    TONE: Professional, secure."""
)


@traceable
@kg_agent.tool 
async def delete_rag_storage(ctx: RunContext[AgentDeps]) -> str:
    """Deletes all data in the RAG storage and Neo4j Graph.
    Use this tool ONLY when the user explicitly requests to "delete", "wipe," "reset," or "clear" the entire system - it is a destructive operation.
    
    Args:
        ctx (RunContext[AgentDeps]): The run context containing dependencies.

    Returns:
        str: Confirmation message upon completion.
    """

    target_file = "kv_store_full_docs.json"
    target_file_doubleCheck = "kv_store_entity_chunks.json"
    
    start_dir = "./"

    file_path = find_file(target_file, start_dir)
    file_path_doubleCheck = find_file(target_file_doubleCheck, start_dir)

    if not file_path:
        LOGGER.error(f"Could not find {target_file} in {start_dir} or its subdirectories.")
        return
    
    if ctx.deps.lightrag:
        try:
            dicts = read_json_file(file_path)
            id_list = [json.dumps(d) for d in dicts]
            
            for id in id_list:
                await ctx.deps.lightrag.adelete_by_doc_id(id.replace('"', ''), True)

            dicts_doubleCheck = read_json_file(file_path_doubleCheck)
            if dicts_doubleCheck:
                entity_chunks = [json.dumps(d) for d in dicts_doubleCheck]

                for entity_name in entity_chunks:
                    await ctx.deps.lightrag.adelete_by_entity(entity_name.replace('"', ''))

            #Clean up LLM query cache
            ct = CleanupTool(ctx.deps.lightrag)
            ct.run()
        except Exception as e:
            LOGGER.error(f"Error during deletion of documents: {e}")
    
    return "RAG storage has been cleared."

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
    lightrag = await initialize_rag(working_dir=WORKING_DIR)
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