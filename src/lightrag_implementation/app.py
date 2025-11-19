import os, asyncio
import streamlit as st
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart

from lightrag.lightrag import LightRAG
from functools import partial
from lightrag.rerank import generic_rerank_api
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_share_data, initialize_pipeline_status

from basic_operations import initialize_rag
from rag_agent import agent, RAGDeps

from dotenv import load_dotenv

import nest_asyncio
nest_asyncio.apply()

load_dotenv()
WORKING_DIR = "./rag_storage"
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

rerunk_func = partial(
    generic_rerank_api, 
    model=os.getenv("RERANK_MODEL"),
    base_url=os.getenv("RERANK_BINDING_HOST"),
    api_key=os.getenv("RERANK_BINDING_API_KEY")
)

async def get_agent_deps() -> RAGDeps:
    """
    Creates a LightRAG instance
    And then uses that to create the Pydantic AI agent dependencies.
    """
    rag = LightRAG(
        working_dir=WORKING_DIR,
        graph_storage="Neo4JStorage",
        llm_model_func=ollama_model_complete,
        llm_model_name="llama3.1:8b",#"qwen2.5:14b",
        llm_model_max_async=4,
        llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
        vector_storage="FaissVectorDBStorage",
        rerank_model_func=rerunk_func,
        min_rerank_score=0.5,
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
    await rag.initialize_storages()
    # Ensure shared dicts exist
    initialize_share_data()

    await initialize_pipeline_status()
    return RAGDeps(lightrag=rag)

def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # user-prompt
    if part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)             

async def run_agent_with_streaming(user_input):
    async with agent.run_stream(
        user_input, deps=st.session_state.agent_deps, message_history=st.session_state.messages
    ) as result:
        async for message in result.stream_text(delta=True):  
            yield message

    # Add the new messages to the chat history (including tool calls and responses)
    st.session_state.messages.extend(result.new_messages())


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~ Main Function with UI Creation ~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

async def main():
    st.title("LightRAG AI Agent")

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = [
            ModelResponse(parts=[TextPart(content= """
### 👋 Hi, I'm your **LightRAG AI Agent**!

I can assist you with:

- 🔍 **Information retrieval** — just ask a question  
- ♻️ **Graph deletion & recreation** — say "recreate graph"  
- 📥 **Data indexing** — provide the path to your *preprocessed* directory  
- 📨 **Data preprocessing** — give me the path to your *.msg* files  
- 🔚 **Exit / stop the agent** — say "exit", "quit", or "stop"

How can I help you today?
""")])
    ] 
        
    if "agent_deps" not in st.session_state:
        st.session_state.agent_deps = await get_agent_deps()  

    # Display all messages from the conversation so far
    # Each message is either a ModelRequest or ModelResponse.
    # We iterate over their parts to decide how to display them.
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    user_input = st.chat_input("What do you want to know?")

    if user_input:
        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            # Create a placeholder for the streaming text
            message_placeholder = st.empty()
            full_response = ""
            
            # Properly consume the async generator with async for
            generator = run_agent_with_streaming(user_input)
            async for message in generator:
                full_response += message
                message_placeholder.markdown(full_response + "▌")
            
            # Final response without the cursor
            message_placeholder.markdown(full_response)


if __name__ == "__main__":
    asyncio.run(main())