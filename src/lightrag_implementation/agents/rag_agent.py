import os, asyncio, shutil
from dataclasses import dataclass
from typing import List
from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from lightrag import LightRAG, QueryParam
from langchain_neo4j import Neo4jGraph
from lightrag_implementation.basic_operations import initialize_rag, index_data
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import argparse

load_dotenv()

WORKING_DIR = "./lightrag_implementation/rag_storage"
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

model = OpenAIChatModel(
    os.getenv("LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8"),
    provider = OpenAIProvider(
        base_url=os.getenv("LLM_BINDING_HOST"), 
        api_key=os.getenv("LLM_BINDING_API_KEY")
    )    
)

class RefinedQueries(BaseModel):
    refined_queries: List[str] = Field(
        description="A list of 1 to 3 highly specific, optimal search queries."
    )
    reasoning: str = Field(
        description="Brief explanation of the thought process behind the refinement."
    )

@dataclass
class RAGDeps:
    """Dependencies for the RAG agent."""
    lightrag: LightRAG
    refinement_llm: ChatOpenAI

# Create the Pydantic AI agent
agent = Agent(
    model,
    deps_type=RAGDeps,
    system_prompt="""You are a Retrieval Expert Agent operating over a LightRAG pipeline to answer user queries accurately.

    You have access to tools: retrieve, rephrase_and_refine_query 
    If you need to use a tool, generate the tool call. Once the tool returns a result, summarize that result for the user.

    1. Retrieval-first: For any information-seeking query, ensure retrieval is performed before answering.
    2. Query refinement: Use `rephrase_and_refine_query` when the user query is ambiguous, incomplete, or overly broad.
    3. Tool usage: When a tool is required, generate the appropriate tool call and summarize the result for the user.
    4. Accuracy: If retrieval returns no results, clearly state that the information is not available in the graph.
    
    TONE: Professional, secure, and fact-based."""
)


@agent.tool
async def retrieve(ctx: RunContext[RAGDeps], question: str) -> str:
    """
    Retrieve relevant documents from the RAG system based on the question.

    Args:
        ctx (RunContext[RAGDeps]): The run context containing dependencies.
        question (str): The user's question to retrieve information for.
    
    Returns:
        str: The retrieved information from the RAG system.
    """
    return await ctx.deps.lightrag.aquery(
        query=question,
        param=QueryParam(mode="mix", enable_rerank=True, include_references=True)
    )

@agent.tool
async def rephrase_and_refine_query(ctx: RunContext[RAGDeps], user_query: str) -> RefinedQueries:
    """
    Analyzes a conversational user query and generates 1-3 optimized, specific, 
    and concise search queries that will maximize retrieval accuracy from the email Knowledge Graph. 
    Use this ONLY when the user's query is complex or ambiguous.

    Args:
        ctx: The run context containing dependencies (LLM).
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

async def run_rag_agent(question: str) -> str:
    """Run the RAG agent to answer a question about Pydantic AI.
    
    Args:
        question: The question to answer.
        
    Returns:
        The agent's response.
    """
    # Create dependencies
    lightrag = await initialize_rag()
    ref_llm = ChatOpenAI(
            temperature=0.2, 
            model=os.getenv("LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8"), 
            base_url=os.getenv("LLM_BINDING_HOST"), 
            api_key=os.getenv("LLM_BINDING_API_KEY")
        )
    deps = RAGDeps(lightrag=lightrag, refinement_llm=ref_llm)
    
    # Run the agent
    result = await agent.run(question, deps=deps)
    
    return result.output

def main():
    """Main function to parse arguments and run the RAG agent."""
    parser = argparse.ArgumentParser(description="This script queries the PydanticAI agent with a user question.")
    parser.add_argument("--question", default="Who is Sofia Stafylaraki?", help="The question to answer")

    args = parser.parse_args()

    # Run the agent
    response = asyncio.run(run_rag_agent(args.question))

    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    main()