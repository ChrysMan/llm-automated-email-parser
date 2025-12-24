import os, operator, asyncio
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable
from typing import Annotated, List, TypedDict
from langgraph_supervisor import create_supervisor
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from lightrag_implementation.basic_operations import initialize_rag
from lightrag_implementation.agents.kg_agent import kg_agent
from lightrag_implementation.agents.rag_agent import rag_agent
from lightrag_implementation.agents.agent_deps import AgentDeps
from lightrag_implementation.tools.preprocessing_tool import execute_full_preprocessing
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "supervisor_agent"

WORKING_DIR = "./lightrag_implementation/rag_storage"
os.makedirs(WORKING_DIR, exist_ok=True)

# # --- State Definition ---
# class State(TypedDict):
#     messages: Annotated[List[BaseMessage], operator.add]
#     Deps: AgentDeps

# --- Tool Definitions ---
def create_tools(deps: AgentDeps):
    """
    This 'factory' function creates tools that have 
    access to the 'deps'.
    """
    @tool
    async def kg_tool(query: str):
        """
        Use this tool to interact with the KG agent for tasks like reinitializing graph storage, adding data, or finalizing pipelines.
        """
        result = await kg_agent.run(query, deps=deps)
        return result.output

    @tool
    async def rag_tool(query: str):
        """
        Use this tool to interact with the RAG agent for retrieving information/answering questions from the system.
        """
        result = await rag_agent.run(query, deps=deps)
        return result.output
    
    return [kg_tool, rag_tool, execute_full_preprocessing]

# --- Supervisor Setup ---
supervisor_prompt = """You are a supervisor overseeing:
1. kg_tool: Use this tool for reinitializing graph storage, adding data to the graph, or finalizing pipelines.
2. rag_tool: Use this tool for retrieving information/answering questions from the system and refining queries. If the user query is ambiguous, incomplete, or overly broad, use rag_tool to refine the query before retrieval.
3. execute_full_preprocessing: Use this tool to preprocess email data from a directory.

Reason over which agent or tool to call based on the user's request and call them appropriately. 
You can call multiple agents/tools in a single turn. Provide clear and detailed responses based on the outputs of the tools you invoke.
Do not include the references of the documents in the final answer. 
Be professional, helpful, and kind in your tone."""

# Run the graph
@traceable
async def run_supervisor():
    lightrag = await initialize_rag(working_dir=WORKING_DIR)
    ref_llm = ChatOpenAI(
        temperature=0.2, 
        model=os.getenv("LLM_MODEL", "gpt-4o"), 
        base_url=os.getenv("LLM_BINDING_HOST"), 
        api_key=os.getenv("LLM_BINDING_API_KEY")
    ) 
    deps = AgentDeps(lightrag=lightrag, refinement_llm=ref_llm)

    tools = create_tools(deps)

    # 2. Initialize the agent WITH these specific tools
    # Since deps are inside the tools, we don't need a custom state_schema!
    supervisor_agent = create_agent(
        model=ChatOpenAI(
            temperature=0, 
            model=os.getenv("LLM_AGENT_MODEL", "gpt-4o"), 
            base_url=os.getenv("LLM_AGENT_BINDING_HOST"), 
            api_key=os.getenv("LLM_AGENT_BINDING_API_KEY"),
            max_retries=3
        ),
        tools=tools,
        system_prompt=supervisor_prompt,
        checkpointer=InMemorySaver()
    )
    
    print("\n--- Arian Email System Active (Type 'exit' to quit) ---")
    config = {"configurable": {"thread_id": "session_1"}}

    while True:
        user_input = input("\n[User]: ").strip()
        
        is_exiting = user_input.lower() in ["exit", "quit", "stop", "bye"]
        
        if is_exiting:
            # Force a finalization message to the supervisor
            print("\n[System]: Finalizing LightRAG pipeline before exit...")
            user_input = "Please finalize the lightrag pipeline without reinitializing."

        inputs = {
            "messages": [HumanMessage(content=user_input)],
            "Deps": deps
        }

        try:
            async for step in supervisor_agent.astream(inputs, config=config, stream_mode="updates"):
                for node_name, update in step.items():
                    if node_name == "supervisor":
                        continue
                    
                    if "messages" in update:
                        for message in update.get("messages", []):
                            #print("\n[System]: ", message.content)
                            print(f"\n--- Output from [{node_name}] ---\n{message.content}")
                            #message.pretty_print()
        except Exception as e:
            print(f"[Error]: An error occurred - {e}")
            continue

        if is_exiting:
            print("\n[System]: Pipeline finalized. Goodbye!")
            break

    

if __name__ == "__main__":
    try:
        asyncio.run(run_supervisor())
    except KeyboardInterrupt:
        pass