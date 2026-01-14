import os, asyncio
from typing import List
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from pydantic_ai.agent import Agent
from pydantic_ai import RunContext

from ..core.llm import agent_llm, ref_llm
from ..core.pipeline import initialize_rag
from ..agents.kg_agent import kg_agent
from ..agents.rag_agent import rag_agent
from ..agents.dependencies import AgentDeps
from ..tools.preprocessing_tool import execute_full_preprocessing
from dotenv import load_dotenv

load_dotenv()

if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "supervisor_agent"

def create_supervisor_agent()-> Agent:
    supervisor_agent = Agent(
        agent_llm,
        deps_type=AgentDeps,
        retries=3,
        model_settings={'parallel_tool_calls': False, 'tool_call_order': ['split_complex_query']},
        system_prompt="""You are an Enterprise supervisor agent that handles a knowledge graph and a retrieval-augmented generation (RAG) pipeline made from company email threads. 
    You are a helper for the employees of a maritime corporation. You coordinate two specialized agents and a tool:
        
    1. split_complex_query: Use this tool when a user request contains more than one action or instruction in order to break it down into separate, atomic tasks.
    2. kg_tool: Use this tool for deleting graph storage, adding data to the graph, or finalizing pipelines.
    3. rag_tool: Use this tool for retrieving information/answering questions from the knowledge graph and refining queries. If the user query is ambiguous, incomplete, or overly broad, use rag_tool to refine the query before retrieval.
    4. preprocess_emails: Use this tool to preprocess .msg files containing email threads from a given directory. If the directory isn't provided, ask for it. Use ONLY when asked to preprocess email data. Don't use it when asked to add data to the graph.

    IMPORTANT: 
    1. Only finalize the RAG pipeline when the user explicitly requests it (e.g., 'finalize the pipeline', 'close', 'exit' etc.).
    2. If the user provides multiple instructions you MUST first call `split_complex_query` to split it into atomic steps. You must then sequentially call the appropriate tools for each atomic step.
    3. When asked to add data assume that the data has already been preprocessed.
    4. When asked to preprocess emails and then add them to the graph, first call `preprocess_emails` and once it is complete, call `kg_tool` to add the preprocessed data to the graph using the same directory.
    
    Rules:
    1. Strategic Routing: Select the appropriate agent or tool based on the user request.
    2. Strict Sequentiality: Execute tools one at a time. Do not initiate a new tool until the previous one has fully responded.
    3. Multi-Step Reasoning: Call multiple tools within a single turn if necessary, but compile the final response only after all operations are complete.
    4. Clean Output: Provide professional, detailed answers based on tool results, but strictly exclude all document references or citations."""
    )

    @supervisor_agent.tool
    async def kg_tool(ctx: RunContext[AgentDeps], query: str) -> str:
        """
        Use this tool to interact with the KG agent for tasks like deleting graph storage, adding data from a directory, or finalizing pipelines.
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
    def preprocess_emails(ctx: RunContext[AgentDeps], dir_path: str) -> str:
        """ Preprocess .msg files that contain email threads from the specified directory and return cleaned and unique email texts. """
        if dir_path is None or dir_path.strip() == "":
            if ctx.deps.dir_path is not None:
                dir_path = ctx.deps.dir_path
            else:
                return "Please provide a valid directory path."
        else:
            ctx.deps.dir_path = dir_path

        if not os.path.isdir(dir_path):
            return f"{dir_path} is not a valid directory."
        elif not any(fname.endswith('.msg') for fname in os.listdir(dir_path)):
            return f"No .msg files found in the directory {dir_path}."
        elif any(fname.endswith('unique.json') for fname in os.listdir(dir_path)):
            return f"Email preprocessing has already been completed for the directory {dir_path}."
        else:
            result = execute_full_preprocessing(dir_path = dir_path)
            return result
        
    class SplitQueries(BaseModel):
        queries: List[str] = Field(description="A list of the individual, atomic requests found in the user's input.")

    @supervisor_agent.tool
    async def split_complex_query(ctx: RunContext[AgentDeps], user_input: str) -> SplitQueries:
        """
        Use this tool when a user provides multiple instructions, commands, or questions.
        It splits the input into a list of separate, standalone tasks or questions. 
        Inside the questions, make sure to clarify any ambiguous references.
        Example: 'Who is Alice and what did she send?' -> ['Who is Alice?', 'What did Alice send?']
        Example: 'Preprocess the data from the directory /data/emails and then add them to the graph.' 
        -> ['Preprocess the data from the directory /data/emails', 'Add data from the directory /data/emails to the graph']
        """
        prompt = f""" Analyze the following user input and determine if it contains multiple distinct questions or requests.
        If it does, split them into a list of standalone, complete sentences. 
        If it is only one request, return it as a single-item list.
        
        USER INPUT: {user_input}
        """
        
        llm_with_structure = ctx.deps.refinement_llm.with_structured_output(SplitQueries)
        result = await llm_with_structure.ainvoke(prompt)
        return result
    
    return supervisor_agent

async def run_supervisor():

    rag = await initialize_rag()

    deps = AgentDeps(
        lightrag=rag,
        refinement_llm=ref_llm
    )

    agent = create_supervisor_agent()

    message_history = []

    while True:
        user_input = input("\n[User]: ")
        is_exiting = user_input.lower() in ["exit", "quit", "stop", "bye"]
        if is_exiting:
            print("\n[System]: Finalizing LightRAG pipeline before exit...")
            user_input = "Please finalize the lightrag pipeline."

        result = await agent.run(user_input, deps=deps, message_history=message_history)
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
