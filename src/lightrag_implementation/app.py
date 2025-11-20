import os, asyncio
import streamlit as st
import nest_asyncio
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart

from basic_operations import initialize_rag
from rag_agent import agent, RAGDeps

from dotenv import load_dotenv

nest_asyncio.apply()

load_dotenv()
WORKING_DIR = "./rag_storage"
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)


def get_agent_deps() -> RAGDeps:
    """
    Creates a LightRAG instance
    And then uses that to create the Pydantic AI agent dependencies.
    """
    rag = asyncio.run(initialize_rag(WORKING_DIR))
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
        async for message in result.stream_text(delta=True, debounce_by=None):  
            yield message

    # Add the new messages to the chat history (including tool calls and responses)
    st.session_state.messages.extend(result.new_messages())


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~ Main Function with UI Creation ~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main():
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
        st.session_state.agent_deps = get_agent_deps()
        
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
            # full_response = ""
            with st.spinner('Thinking...'):
                response = agent.run_sync(user_input, deps=st.session_state.agent_deps, message_history=st.session_state.messages)
                message_placeholder.markdown(response.output)
            # # Properly consume the async generator with async for
            # async def stream_reply():
            #     async for delta in run_agent_with_streaming(user_input):
            #         nonlocal full_response
            #         full_response += delta
            #         message_placeholder.markdown(full_response + "▌")

            #     message_placeholder.markdown(full_response)
            
            # # Final response without the cursor
            # message_placeholder.markdown(full_response)

        #asyncio.run(stream_reply())


if __name__ == "__main__":
    main()
    