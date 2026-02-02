import os, asyncio
from typing import List
from pydantic import BaseModel, Field
from pydantic_ai.agent import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai import RunContext

from ..core.llm import agent_llm, ref_llm
from ..core.pipeline import initialize_rag
from ..agents.kg_agent import kg_agent
from ..agents.rag_agent import rag_agent
from ..agents.dependencies import AgentDeps
from ..tools.preprocessing_tool import execute_full_preprocessing
from dotenv import load_dotenv

load_dotenv()

# if os.getenv("LANGSMITH_API_KEY"):
#     os.environ["LANGCHAIN_TRACING_V2"] = "true"
#     os.environ["LANGCHAIN_PROJECT"] = "supervisor_agent"

MAX_HISTORY_LENGTH = 4

async def history_processor(history: list[ModelMessage])-> list[ModelMessage]:
    """Trim message history to prevent memory issues in long conversations."""
    if len(history) <= MAX_HISTORY_LENGTH:
        return history
    
    return history[-MAX_HISTORY_LENGTH:]

def create_supervisor_agent()-> Agent:
    supervisor_agent = Agent(
        agent_llm,
        deps_type=AgentDeps,
        retries=3,
        history_processors=[history_processor],
        model_settings={'parallel_tool_calls': False, 'tool_call_order': ['split_complex_query']},
        system_prompt="""You are an Enterprise Supervisor Agent managing a knowledge graph and RAG pipeline for maritime corporation email threads. You coordinate specialized agents and tools to assist employees.

TOOLS:
1. split_complex_query: Break down multi-action requests into atomic tasks before processing.
2. rag_tool: Retrieve information, answer questions, and refine ambiguous/incomplete queries from the knowledge graph.
3. kg_tool: Handle graph operations - add data, delete storage, clear cache, or finalize pipelines.
4. preprocess_emails: Clean and extract email threads from .msg files in specified directories.

CRITICAL GUIDELINES:
1. Pipeline Finalization: Only finalize/close when explicitly requested (e.g., 'finalize', 'close', 'exit').
2. Complex Queries: ALWAYS use split_complex_query first for multi-instruction requests, then execute tools sequentially.
3. Exclusive Usage: Use kg_tool ONLY for graph data operations; preprocess_emails ONLY for email preprocessing.
4. Sequential Execution: Process one tool at a time; complete all operations before final response.

OPERATIONAL PROTOCOLS:
1. Context Awareness: Consider full conversation history for coherent responses.
2. Smart Routing: Select appropriate tools based on request analysis.
3. Response Quality: Do not make questions to the user. Provide professional, detailed answers strictly based on tool results. Answer only what is explicitly asked. Do not include citations or additional information.
"""
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
        """ Use this tool to preprocess .msg files that contain email threads from the specified directory and return cleaned and unique email texts. """
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
