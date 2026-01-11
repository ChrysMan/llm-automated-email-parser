import os, asyncio
from typing import List
from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from lightrag import QueryParam
from dotenv import load_dotenv

from ..core.pipeline import initialize_rag
from ..core.llm import ref_llm, agent_llm
from ..agents.dependencies import AgentDeps
from utils.logging import LOGGER
from utils.file_io import find_dir

load_dotenv()

langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
if langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
    os.environ["LANGSMITH_PROJECT"] = "rag_agent"
else:
    LOGGER.warning("Langsmith API key not found. Tracing will be disabled.")

WORKING_DIR = find_dir("rag_storage", "./")
os.makedirs(WORKING_DIR, exist_ok=True)

class RefinedQueries(BaseModel):
    refined_queries: List[str] = Field(
        description="A list of 1 to 3 highly specific, optimal search queries."
    )
    reasoning: str = Field(
        description="Brief explanation of the thought process behind the refinement."
    )

# Create the Pydantic AI agent
rag_agent = Agent(
    agent_llm,
    deps_type=AgentDeps,
    retries=3,
    model_settings={'parallel_tool_calls': False},
    system_prompt="""You are an Enterprise Retrieval Expert Agent operating over a LightRAG pipeline to answer user queries accurately. 
    The Knowledge Graph contains email data from a maritime corporation's internal and external communications.

    You have access to tools: retrieve, rephrase_and_refine_query 

    OPERATIONAL RULES:
    1. Retrieval-first: For any information-seeking query, ensure retrieval is performed from the knowledge graph before answering.
    2. Query refinement: Use `rephrase_and_refine_query` when the user query is ambiguous, incomplete, complex, or overly broad and then use the refined queries to retrieve relevant documents from the knowledge graph. Use this tool instead of questioning the user for clarification.
    3. Tool usage: Reason over which tool to call based on the user's request and call them appropriately. 
    You can call multiple tools in a single turn sequencially until the question is answered.
    4. Accuracy: Answer only information relevant to the question asked. If retrieval returns no results, clearly state that the information is not available in the graph.
    5. Output: Provide clear and detailed responses based on the outputs of the tools you invoke.

    TONE: Professional, secure, and fact-based."""
)


@rag_agent.tool
async def retrieve(ctx: RunContext[AgentDeps], question: str) -> str:
    """
    Retrieve relevant documents from the RAG system based on the question.

    Args:
        ctx (RunContext[AgentDeps]): The run context containing dependencies.
        question (str): The user's question to retrieve information for.
    
    Returns:
        str: The retrieved information from the RAG system.
    """
    return await ctx.deps.lightrag.aquery(
        query=question,
        param=QueryParam(mode="mix", enable_rerank=True, include_references=False),
        #system_prompt="""Project Integrity Rule: Every entity is bound to a specific Project Reference Number found in its file_path (e.g., '244036') and in the description. When answering a query about a specific project, you must filter the retrieved entities by this reference number."""
    )

@rag_agent.tool
async def rephrase_and_refine_query(ctx: RunContext[AgentDeps], user_query: str) -> RefinedQueries:
    """
    Use this ONLY when the user's query is complex, ambiguous, or overly simple.
    Analyzes a conversational user query and generates 1-3 optimized, specific, 
    and concise search queries that will maximize retrieval accuracy from the email Knowledge Graph. 

    Args:
        ctx (RunContext[AgentDeps]): The run context containing dependencies (LLM).
        user_query: The conversational query from the user.

    Returns:
        RefinedQueries: A Pydantic model containing the list of refined queries and reasoning.
    """
    try:
        template = f"""
        You are an expert Query Refinement Agent. Your task is to analyze the user's query
        about email data and generate 1 to 3 highly specific, concise queries that will maximize retrieval 
        accuracy from a knowledge graph .

        The queries should focus on key entities, dates, and topics found in the user's input.
        
        CONVERSATIONAL QUERY: {user_query}
        """
        
        llm_with_structure = ctx.deps.refinement_llm.with_structured_output(RefinedQueries)
        
        result = await llm_with_structure.ainvoke(template)
        return result

    except Exception as e:
        return RefinedQueries(
            refined_queries=[],
            reasoning=f"Technical failure during async refinement: {str(e)}"
        )

async def run_interactive_loop():
    """Starts a continuous chat session with the KG Agent."""
    lightrag = await initialize_rag(working_dir=WORKING_DIR)

    deps = AgentDeps(lightrag=lightrag, refinement_llm=ref_llm)

    message_history = []
    
    print("\n--- RAG Agent Interactive Session (Type 'exit' to quit) ---")

    while True:
        user_input = input("\nUser: ").strip()
        print("\n")
        
        if user_input.lower() in ["exit", "quit", "stop", "bye"]:
            await rag_agent.run("Close the pipeline and exit.", deps=deps, message_history=message_history)
            print("Goodbye!")
            break

        if not user_input:
            continue

        try:
            result = await rag_agent.run(
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