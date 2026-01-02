import os, asyncio
from langchain_openai import ChatOpenAI
import streamlit as st
import nest_asyncio
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart
from functools import partial
from lightrag.lightrag import LightRAG
from lightrag.rerank import generic_rerank_api
from lightrag.llm.ollama import ollama_embed
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_share_data, initialize_pipeline_status

from lightrag_implementation.basic_operations import initialize_rag
from lightrag_implementation.agents.agent_deps import AgentDeps, LightRAGBox
from lightrag_implementation.multiagent_pydantic import supervisor_agent 
from utils.graph_utils import find_dir
from dotenv import load_dotenv

nest_asyncio.apply()

if 'GLOBAL_LOOP' not in globals():
    GLOBAL_LOOP = asyncio.new_event_loop()
    asyncio.set_event_loop(GLOBAL_LOOP)

load_dotenv()

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

@st.cache_resource
def get_cached_agent_deps():
    # We need a bridge because st.cache_resource is synchronous 
    # but your setup is asynchronous.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Run the async initialization in this persistent loop
    deps = loop.run_until_complete(get_agent_deps())
    return deps

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
    async with supervisor_agent.run_stream(
        user_input, deps=st.session_state.agent_deps, message_history=st.session_state.messages
    ) as result:
        async for message in result.stream_text(delta=True, debounce_by=None):  
            yield message

    # Add the new messages to the chat history (including tool calls and responses)
    st.session_state.messages = result.all_messages()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~ Main Function with UI Creation ~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

async def main():
    st.title("Arian AI Agent")


    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []
#         st.session_state.messages = [
#             ModelResponse(parts=[TextPart(content= """
# ### 👋 Hi, I'm your **Arian AI Agent**!

# I can assist you with:

# - 🔍 **Information retrieval** — just ask a question  
# - ♻️ **Graph deletion** — say "delete graph"  
# - 📥 **Data indexing** — provide the path to your *preprocessed* directory  
# - 📨 **Data preprocessing** — give me the path to your *.msg* files  
# - 🔚 **Exit / stop the agent** — say "exit", "quit", or "stop"

# How can I help you today?
# """)])
#     ] 
        
    if "agent_deps" not in st.session_state:
        with st.spinner("Initializing knowledge graph and agents..."):
            st.session_state.agent_deps = GLOBAL_LOOP.run_until_complete(get_agent_deps())
            st.success("System Ready!")
        #st.session_state.agent_deps = get_cached_agent_deps()

    st.session_state.agent_deps.lightrag.loop = asyncio.get_running_loop()
        
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
            
            # with st.spinner('Thinking...'):
            #     response = supervisor_agent.run_sync(user_input, deps=st.session_state.agent_deps, message_history=st.session_state.messages)
            #     message_placeholder.markdown(response.output)

            # st.session_state.messages.extend(response.all_messages())
            #st.rerun()
            


if __name__ == "__main__":
    GLOBAL_LOOP.run_until_complete(main())
    