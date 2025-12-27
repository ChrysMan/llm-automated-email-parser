import os, asyncio
from typing import List
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from pydantic_ai.agent import Agent
from pydantic_ai import RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from lightrag_implementation.basic_operations import initialize_rag
from lightrag_implementation.agents.kg_agent import kg_agent
from lightrag_implementation.agents.rag_agent import rag_agent
from lightrag_implementation.agents.agent_deps import AgentDeps, LightRAGBox
from lightrag_implementation.tools.preprocessing_tool import execute_full_preprocessing
from utils.graph_utils import find_dir
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "supervisor_agent"

WORKING_DIR = find_dir("rag_storage", "./")
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
    system_prompt="""You are an Enterprise supervisor agent that handles a knowledge graph and a retrieval-augmented generation (RAG) pipeline made from company email threads. 
You are a helper for the employees of a maritime corporation. You coordinate two specialized agents and a tool:
    
1. split_complex_query: Use this tool when a user requests multiple things in one question. It is usually used first.
2. kg_tool: Use this tool for deleting graph storage, adding data to the graph, or finalizing pipelines.
3. rag_tool: Use this tool for retrieving information/answering questions from the knowledge graph and refining queries. If the user query is ambiguous, incomplete, or overly broad, use rag_tool to refine the query before retrieval.
4. preprocess_emails: Use this tool to preprocess .msg files containing email threads from a given directory. If the directory isn't provided, ask for it.

Reason over which agent or tool to call based on the user's request and call them appropriately. 
You can call multiple agents/tools in a single turn. Provide clear and detailed responses based on the outputs of the tools you invoke.
Do not include the references of the documents in the final answer. 
Be professional, helpful, and kind in your tone."""
)

@supervisor_agent.tool
async def kg_tool(ctx: RunContext[AgentDeps], query: str) -> str:
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
def preprocess_emails(ctx: RunContext[AgentDeps], directory_path: str) -> str:
    """ Preprocess .msg files that contain email threads from the specified directory and return cleaned and unique email texts. """
    if directory_path is None or directory_path.strip() == "":
        return "Please provide a valid directory path."
    elif not os.path.isdir(directory_path):
        return f"{directory_path} is not a valid directory."
    else:
        result = execute_full_preprocessing(dir_path = directory_path)
        return result
    
class SplitQueries(BaseModel):
    queries: List[str] = Field(description="A list of the individual, atomic requests found in the user's input.")

@supervisor_agent.tool
async def split_complex_query(ctx: RunContext[AgentDeps], user_input: str) -> SplitQueries:
    """
    Use this tool when a user provides multiple instructions, commands, or questions.
    It splits the input into a list of separate, standalone tasks or questions.
    Example: 'Who is Alice and what did she send?' -> ['Who is Alice?', 'What did Alice send?']
    Example: 'Find the invoice and then delete the source folder' 
    -> ['Find the invoice', 'Delete the source folder']
    """
    prompt = f""" Analyze the following user input and determine if it contains multiple distinct questions or requests.
    If it does, split them into a list of standalone, complete sentences. 
    If it is only one request, return it as a single-item list.
    
    USER INPUT: {user_input}
    """
    
    llm_with_structure = ctx.deps.refinement_llm.with_structured_output(SplitQueries)
    result = await llm_with_structure.ainvoke(prompt)
    return result

async def run_supervisor():

    lightrag = await initialize_rag(working_dir=WORKING_DIR)

    ref_llm = ChatOpenAI(
        temperature=0.2, 
        model=os.getenv("LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8"), 
        base_url=os.getenv("LLM_BINDING_HOST"), 
        api_key=os.getenv("LLM_BINDING_API_KEY")
    )

    rag_box = LightRAGBox(instance=lightrag)

    deps = AgentDeps(
        rag_box=rag_box,
        refinement_llm=ref_llm
    )

    message_history = []

    while True:
        user_input = input("\n[User]: ")
        is_exiting = user_input.lower() in ["exit", "quit", "stop", "bye"]
        if is_exiting:
            print("\n[System]: Finalizing LightRAG pipeline before exit...")
            user_input = "Please finalize the lightrag pipeline."

        result = await supervisor_agent.run(user_input, deps=deps, message_history=message_history)
        print(f"\n[System]]: {result.output}")

        message_history = result.all_messages()

        if is_exiting:
            print("\n[System]: Pipeline finalized. Goodbye!")
            break

if __name__ == "__main__":
    try:
        asyncio.run(run_supervisor())
    except KeyboardInterrupt:
        pass
