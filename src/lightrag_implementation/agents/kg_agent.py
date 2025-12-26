import os, asyncio, json
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from lightrag_implementation.basic_operations import initialize_rag, index_data
from lightrag_implementation.agents.agent_deps import AgentDeps
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from langsmith import traceable
from dotenv import load_dotenv
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

model = OpenAIChatModel(
    os.getenv("LLM_AGENT_MODEL", "cyankiwi/Ministral-3-8B-Reasoning-2512-AWQ-8bit"),
    provider = OpenAIProvider(
        base_url=os.getenv("LLM_AGENT_BINDING_HOST"), 
        api_key=os.getenv("LLM_AGENT_BINDING_API_KEY")
    )    
)

# Create the Pydantic AI agent
kg_agent = Agent(
    model,
    deps_type=AgentDeps,
    model_settings={'parallel_tool_calls': False},
    system_prompt="""You are an Enterprise Email Intelligence Agent managing a LightRAG pipeline and Neo4j Knowledge Graph.

    You have access to tools: add_data, delete_rag_storage, and close.
    Reason over which tool to call based on the user's request and call them appropriately. 
    You can call multiple tools in a single turn sequencially until the question is answered. Provide clear and detailed responses based on the outputs of the tools you invoke.

    OPERATIONAL RULES:
    1. Data Integrity: Only use `add_data` when provided a directory path. Verify the existence of the directory before proceeding.
    2. Safety: `delete_rag_storage` is destructive. Ask for confirmation before using it.
    3. Finalization: You MUST call `close` when the user requests to close or finalize the system to ensure data is saved and connections are closed safely.
    4. Wait for a confirmation message from one tool before calling the next.

    TONE: Professional, secure."""
)


@traceable
@kg_agent.tool 
async def delete_rag_storage(ctx: RunContext[AgentDeps]) -> str:
    """Deletes all data in the RAG storage and Neo4j Graph.
    Use this tool ONLY when the user explicitly requests to "wipe," "reset," or "clear" the entire system - it is a destructive operation.
    
    Args:
        ctx (RunContext[AgentDeps]): The run context containing dependencies.

    Returns:
        str: Confirmation message upon completion.
    """

    target_file = "kv_store_full_docs.json"
    start_dir = "./"

    file_path = find_file(target_file, start_dir)

    if not file_path:
        LOGGER.error(f"Could not find {target_file} in {start_dir} or its subdirectories.")
        return
    
    if ctx.deps.lightrag:
        try:
            dicts = read_json_file(file_path)
            id_list = [json.dumps(d) for d in dicts]
            
            for id in id_list:
                await ctx.deps.lightrag.adelete_by_doc_id(id.replace('"', ''), True)
        except Exception as e:
            LOGGER.error(f"Error during deletion of documents: {e}")
    
    return "RAG storage has been cleared."

@traceable
@kg_agent.tool
async def add_data(ctx: RunContext[AgentDeps], dir_path: str) -> str:
    """Adds data from the given file path into the RAG system.
    
    Args:
        ctx (RunContext[AgentDeps]): The run context containing dependencies.
        dir_path (str): The directory path containing data files to index.

    Returns:
        str: Confirmation message upon completion or Error message upon failure.
    """
    if not os.path.isdir(dir_path):
        return f"{dir_path} is not a valid directory."
    return await index_data(ctx.deps.lightrag, dir_path)

@traceable
@kg_agent.tool
async def close(ctx: RunContext[AgentDeps]) -> str:
    """
    Safely finalize and close the LightRAG pipeline when instructed to "exit", "quit", "close" or "stop".

    Args:
        ctx (RunContext[AgentDeps]): The run context containing dependencies.

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

async def run_interactive_loop():
    """Starts a continuous chat session with the KG Agent."""
    # 1. Initialize dependencies once at the start
    lightrag = await initialize_rag(working_dir=WORKING_DIR)
    deps = AgentDeps(lightrag=lightrag)
    
    # 2. Maintain message history for memory
    message_history = []
    
    print("\n--- KG Agent Interactive Session (Type 'exit' to quit) ---")

    while True:
        # 3. Get user input
        user_input = input("\nUser: ").strip()
        print("\n")
        
        if user_input.lower() in ["exit", "quit", "stop", "bye"]:
            # Ensure we call the close tool before exiting
            await kg_agent.run("Close the pipeline and exit.", deps=deps, message_history=message_history)
            print("Goodbye!")
            break

        if not user_input:
            continue

        try:
            # 4. Run the agent with context
            # Pass message_history so the agent remembers previous steps
            result = await kg_agent.run(
                user_input, 
                deps=deps, 
                message_history=message_history
            )
            
            # 5. Update history and show response
            message_history = result.all_messages()
            print(f"\nAgent: {result.output}")

        except Exception as e:
            print(f"\n[SYSTEM ERROR]: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(run_interactive_loop())
    except KeyboardInterrupt:
        print("\nSession interrupted.")