import os, operator, asyncio
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from typing import Annotated, List, TypedDict
from pydantic_ai.agent import Agent
from pydantic_ai import RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider



from lightrag_implementation.basic_operations import initialize_rag
from lightrag_implementation.agents.kg_agent import kg_agent
from lightrag_implementation.agents.rag_agent import rag_agent
from lightrag_implementation.agents.agent_deps import AgentDeps, LightRAGBox
from lightrag_implementation.tools.preprocessing_tool import execute_full_preprocessing
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "supervisor_agent"

WORKING_DIR = "./lightrag_implementation/rag_storage"
os.makedirs(WORKING_DIR, exist_ok=True)

model = OpenAIChatModel(
    os.getenv("LLM_AGENT_MODEL", "cyankiwi/Ministral-3-8B-Reasoning-2512-AWQ-8bit"),
    provider = OpenAIProvider(
        base_url=os.getenv("LLM_AGENT_BINDING_HOST"), 
        api_key=os.getenv("LLM_AGENT_BINDING_API_KEY")
    )    
)

supervisor_agent = Agent(
    model,
    deps_type=AgentDeps,
    retries=3,
    model_settings={'parallel_tool_calls': False},
    system_prompt="""You are a supervisor overseeing:
1. kg_tool: Use this tool for reinitializing graph storage, adding data to the graph, or finalizing pipelines.
2. rag_tool: Use this tool for retrieving information/answering questions from the system and refining queries. If the user query is ambiguous, incomplete, or overly broad, use rag_tool to refine the query before retrieval.
3. execute_full_preprocessing: Use this tool to preprocess email data from a directory.

Reason over which agent or tool to call based on the user's request and call them appropriately. 
You can call multiple agents/tools in a single turn. Provide clear and detailed responses based on the outputs of the tools you invoke.
Do not include the references of the documents in the final answer. 
Be professional, helpful, and kind in your tone."""
)

@supervisor_agent.tool
async def kg_tool(ctx: RunContext[AgentDeps],query: str) -> str:
    """
    Use this tool to interact with the KG agent for tasks like reinitializing graph storage, adding data, or finalizing pipelines.
    """
    result = await kg_agent.run(query, deps=ctx.deps)
    return result.output

@supervisor_agent.tool
async def rag_tool(ctx: RunContext[AgentDeps], query: str) -> str:
    """
    Use this tool to interact with the RAG agent for retrieving information/answering questions from the system and refining ambiguous queries.
    """
    result = await rag_agent.run(query, deps=ctx.deps)
    return result.output

@supervisor_agent.tool
async def execute_full_preprocessing(ctx: RunContext[AgentDeps], directory_path: str) -> str:
    """
    Preprocess email data from the specified directory and return cleaned and unique email texts."""
    result = await execute_full_preprocessing(directory_path)
    return result

async def run_supervisor():
    lightrag = await initialize_rag(working_dir=WORKING_DIR)
    ref_llm = ChatOpenAI(
        temperature=0.2, 
        model=os.getenv("LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8"), 
        base_url=os.getenv("LLM_BINDING_HOST"), 
        api_key=os.getenv("LLM_BINDING_API_KEY")
    )

    rag_box = LightRAGBox(lightrag)
    
    deps = AgentDeps(
        rag_box=rag_box,
        refinement_llm=ref_llm
    )
    while True:
        user_input = input("\n[User]: ")
        is_exiting = user_input.lower() in ["exit", "quit", "stop", "bye"]
        if is_exiting:
            print("\n[System]: Finalizing LightRAG pipeline before exit...")
            user_input = "Please finalize the lightrag pipeline without reinitializing."

        result = await supervisor_agent.run(user_input, deps=deps)
        print(f"\n[System]]: {result.output}")

        if is_exiting:
            print("\n[System]: Pipeline finalized. Goodbye!")
            break

if __name__ == "__main__":
    try:
        asyncio.run(run_supervisor())
    except KeyboardInterrupt:
        pass
