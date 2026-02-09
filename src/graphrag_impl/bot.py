import streamlit as st
from utils import write_message
from agent import generate_response

from streamlit.runtime.scriptrunner import get_script_run_ctx


st.set_page_config("Ebert", page_icon="🎙️")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm the GraphRAG Chatbot!  How can I help you?"},
    ]

# Submit handler
def handle_submit(message):
    """
    Submit handler:

    You will modify this method to talk with an LLM and provide
    context using data from Neo4j.
    """

    # Handle the response
    with st.spinner('Thinking...'):
        # Call the agent
        response = generate_response(message)
        write_message('assistant', response)

for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

if prompt := st.chat_input("What is up?"):
    write_message('user', prompt)

    handle_submit(prompt)
