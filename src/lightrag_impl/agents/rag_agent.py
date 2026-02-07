import asyncio
from typing import List
from lightrag.lightrag import QueryParam
from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from dotenv import load_dotenv

from ..core.pipeline import initialize_rag, run_async_query
from ..core.llm import ref_llm, agent_llm
from ..agents.dependencies import AgentDeps
from utils.logging import LOGGER

load_dotenv()

# if os.getenv("LANGSMITH_API_KEY"):
#     os.environ["LANGCHAIN_TRACING_V2"] = "true"
#     os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
#     os.environ["LANGSMITH_PROJECT"] = "rag_agent"
# else:
#     LOGGER.warning("Langsmith API key not found. Tracing will be disabled.")


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
    system_prompt="""You are an Enterprise Retrieval Expert Agent managing a LightRAG pipeline for maritime corporation email data queries.

TOOLS:
1. retrieve: Search and retrieve relevant information from the knowledge graph.
2. rephrase_and_refine_query: Transform ambiguous, incomplete, or broad queries into 1-3 specific search queries.

CRITICAL GUIDELINES:
1. Retrieval Priority: Always perform retrieval from the knowledge graph before answering information queries.
2. Query Optimization: Use rephrase_and_refine_query for ambiguous/complex/broad queries instead of asking for clarification.
3. Sequential Processing: Execute tools sequentially within a single turn when needed to fully answer questions.
4. Accuracy Focus: Answer only based on retrieved information; clearly state when data is unavailable.
5. Response Quality: Provide a comprehensive and detailed answer derived strictly from tool outputs. 
Ensure the response remains exclusively focused on the specific question asked; avoid providing peripheral context, unsolicited advice, or any information not directly required to answer the query.

COMMUNICATION: Maintain professional, secure, and evidence-based responses."""
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
    response = await run_async_query(rag = ctx.deps.lightrag, question=question, mode="mix")  
    #print(f"{type(response)}, {type(response.get('llm_response', ''))}")  # Debugging line to check the structure of the response
    return response.get("llm_response", "")

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
    lightrag = await initialize_rag()

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