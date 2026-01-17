import os
import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()

FASTAPI_URL = os.getenv("LIGHTRAG_API_URL")

MAX_HISTORY_LENGTH = 10

def trim_message_history(messages):
    """Trim message history to prevent memory issues in long conversations."""
    if len(messages) <= MAX_HISTORY_LENGTH:
        return messages
    
    return messages[-MAX_HISTORY_LENGTH:]

st.set_page_config(page_title="Arian AI Chat", page_icon="🤖")
st.title("Arian AI Chat")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message here..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        last_messages = trim_message_history(st.session_state["messages"])
        with st.spinner('Thinking...'):
            try:
                response = requests.post(
                    FASTAPI_URL, 
                    json={"message": prompt, "message_history": last_messages}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    assistant_response = data.get("response", "")
                else:
                    assistant_response = f"Error: {response.status_code} - {response.text}"
            
            except Exception as e:
                assistant_response = f"Connection Error: {str(e)}"

        for chunk in assistant_response.split():
            full_response += chunk + " "
            message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    st.rerun()