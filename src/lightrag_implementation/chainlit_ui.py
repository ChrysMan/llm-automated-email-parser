import os
import asyncio
import chainlit as cl
import os, asyncio
from langchain_openai import ChatOpenAI
from functools import partial
from lightrag.lightrag import LightRAG
from lightrag.rerank import generic_rerank_api
from lightrag.llm.ollama import ollama_embed
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_share_data, initialize_pipeline_status

from lightrag_implementation.basic_operations import initialize_rag
from lightrag_implementation.agents.agent_deps import AgentDeps
from lightrag_implementation.multiagent_pydantic import supervisor_agent 
from utils.graph_utils import find_dir
from dotenv import load_dotenv

# Import your existing logic
from lightrag_implementation.agents.agent_deps import AgentDeps
from lightrag_implementation.multiagent_pydantic import supervisor_agent

load_dotenv()

GLOBAL_DEPS_HOLDER = []

MAX_HISTORY_LENGTH = 5

rerunk_func = partial(
    generic_rerank_api, 
    model=os.getenv("RERANK_MODEL"),
    base_url=os.getenv("RERANK_BINDING_HOST"),
    api_key=os.getenv("RERANK_BINDING_API_KEY")
)

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        os.getenv("LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BINDING_HOST"),
        **kwargs,
    )

async def get_agent_deps() -> AgentDeps:
    """
    Creates a LightRAG instance
    And then uses that to create the Pydantic AI agent dependencies.
    """

    WORKING_DIR = find_dir("rag_storage", "./")

    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)

    # loop = asyncio.get_event_loop()
    # inst = loop.run_until_complete(initialize_rag(WORKING_DIR))
    inst = LightRAG(
        working_dir=WORKING_DIR,
        graph_storage="Neo4JStorage",
        llm_model_func=llm_model_func, #ollama_model_complete,
        #llm_model_name="qwen2.5:14b", #"Piyush20/Qwen_14B_Quantized", #"qwen2.5:14b", #"llama3.1:8b",
        llm_model_max_async=4,
        #llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
        vector_storage="FaissVectorDBStorage",
        rerank_model_func=rerunk_func,
        min_rerank_score=0.4,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts, embed_model="nomic-embed-text", host="http://localhost:11434"
                )
        ),
        enable_llm_cache=False
    )

    # Initialize database connections
    await inst.initialize_storages()
    # Ensure shared dicts exist
    initialize_share_data()

    await initialize_pipeline_status()

    # rag_box = LightRAGBox(instance = inst)

    ref_llm = ChatOpenAI(
        temperature=0.2, 
        model=os.getenv("LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8"), 
        base_url=os.getenv("LLM_BINDING_HOST"), 
        api_key=os.getenv("LLM_BINDING_API_KEY")
    )
    return AgentDeps(lightrag=inst, refinement_llm=ref_llm)

def trim_message_history(messages):
    """Trim message history to prevent memory issues in long conversations."""
    if len(messages) <= MAX_HISTORY_LENGTH:
        return messages
    
    # Keep the most recent messages
    return messages[-MAX_HISTORY_LENGTH:]

@cl.on_chat_start
async def start():
    """
    Runs when a user opens the chat. 
    Initializes the RAG and Agent dependencies once and stores them in the session.
    """
    try:
        # Step 1: Initialize the RAG system (Neo4j connections, storage, etc.)
        # Since this is an async function in an async environment, no more loop errors!
        deps = await get_agent_deps()
        GLOBAL_DEPS_HOLDER.append(deps)
        # Step 2: Store in the user session so it persists
        cl.user_session.set("deps", deps)
        cl.user_session.set("history", [])

        await cl.Message(content="🛡️ **Arian AI System Initialized.** Knowledge Graph is active.").send()
    except Exception as e:
        await cl.Message(content=f"❌ Error during initialization: {str(e)}").send()

@cl.on_message
async def main(message: cl.Message):
    """
    Runs every time the user sends a message.
    """
    # Retrieve the persistent dependencies and history
    deps = cl.user_session.get("deps")
    history = cl.user_session.get("history")

    # Prepare an empty message to stream into
    msg = cl.Message(content="")

    try:
        # Run the supervisor agent in streaming mode
        async with supervisor_agent.run_stream(
            message.content, 
            deps=deps, 
            message_history=history
        ) as result:
            # Stream the text chunks as they arrive
            async for chunk in result.stream_text(delta=True):
                await msg.stream_token(chunk)

        # Update the message history in the UI
        await msg.update()

        # Update the conversation history with the full exchange
        history.extend(result.all_messages())
        history = trim_message_history(history)

        cl.user_session.set("history", history)
        

    except Exception as e:
        await cl.ErrorMessage(content=f"Agent Error: {str(e)}").send()