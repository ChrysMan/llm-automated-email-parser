import os, asyncio
from dataclasses import dataclass
from typing import List
from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from lightrag import QueryParam
from lightrag_implementation.basic_operations import initialize_rag
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from langchain_openai import ChatOpenAI
from lightrag_implementation.agents.agent_deps import AgentDeps
from dotenv import load_dotenv
from utils.logging_config import LOGGER

load_dotenv()

langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
if langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
    os.environ["LANGSMITH_PROJECT"] = "rag_agent"
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

class RefinedQueries(BaseModel):
    refined_queries: List[str] = Field(
        description="A list of 1 to 3 highly specific, optimal search queries."
    )
    reasoning: str = Field(
        description="Brief explanation of the thought process behind the refinement."
    )

# Create the Pydantic AI agent
rag_agent = Agent(
    model,
    deps_type=AgentDeps,
    model_settings={'parallel_tool_calls': False},
    system_prompt="""You are an Enterprise Retrieval Expert Agent operating over a LightRAG pipeline to answer user queries accurately. 
    The Knowledge Graph contains email data from a maritime corporation's internal and external communications.

    You have access to tools: retrieve, rephrase_and_refine_query 
    Reason over which tool to call based on the user's request and call them appropriately. 
    You can call multiple tools in a single turn sequencially until the question is answered. Provide clear and detailed responses based on the outputs of the tools you invoke.

    OPERATIONAL RULES:
    1. Retrieval-first: For any information-seeking query, ensure retrieval is performed before answering.
    2. Query refinement: Use `rephrase_and_refine_query` when the user query is ambiguous, incomplete, or overly broad and then use the refined queries to retrieve relevant documents. Use the tool instead of questioning the user for clarification.
    3. Tool usage: When a tool is required, generate the appropriate tool call and summarize the result for the user.
    4. Accuracy: If retrieval returns no results, clearly state that the information is not available in the graph.
    
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
        param=QueryParam(mode="mix", enable_rerank=True, include_references=True)
    )

@rag_agent.tool
async def rephrase_and_refine_query(ctx: RunContext[AgentDeps], user_query: str) -> RefinedQueries:
    """
    Analyzes a conversational user query and generates 1-3 optimized, specific, 
    and concise search queries that will maximize retrieval accuracy from the email Knowledge Graph. 
    Use this ONLY when the user's query is complex or ambiguous.

    Args:
        ctx (RunContext[AgentDeps]): The run context containing dependencies (LLM).
        user_query: The conversational query from the user.

    Returns:
        RefinedQueries: A Pydantic model containing the list of refined queries and reasoning.
    """
    try:
        # Define the local prompt
        template = f"""
        You are an expert Query Refinement Agent. Your task is to analyze the user's query
        about email data and generate 1 to 3 highly specific, concise queries that will maximize retrieval 
        accuracy from a vector database.

        The queries should focus on key entities, dates, and topics found in the user's input.
        
        CONVERSATIONAL QUERY: {user_query}
        """
        
        # Use the LLM from dependencies to get structured output
        # Since we are in an async tool, we use ainvoke
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
    # 1. Initialize dependencies once at the start
    lightrag = await initialize_rag(working_dir=WORKING_DIR)
    ref_llm = ChatOpenAI(
            temperature=0.2, 
            model=os.getenv("LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8"), 
            base_url=os.getenv("LLM_BINDING_HOST"), 
            api_key=os.getenv("LLM_BINDING_API_KEY")
        )
    deps = AgentDeps(lightrag=lightrag, refinement_llm=ref_llm)
    
    # 2. Maintain message history for memory
    message_history = []
    
    print("\n--- RAG Agent Interactive Session (Type 'exit' to quit) ---")

    while True:
        # 3. Get user input
        user_input = input("\nUser: ").strip()
        print("\n")
        
        if user_input.lower() in ["exit", "quit", "stop", "bye"]:
            # Ensure we call the close tool before exiting
            await rag_agent.run("Close the pipeline and exit.", deps=deps, message_history=message_history)
            print("Goodbye!")
            break

        if not user_input:
            continue

        try:
            # 4. Run the agent with context
            # Pass message_history so the agent remembers previous steps
            result = await rag_agent.run(
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