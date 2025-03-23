import extract_msg
from utils.logging_config import LOGGER

def write_file(content, filename):
    # Writes the given content as a text file to the local directory
    with open(f"{filename}.txt", "w") as f:
        f.write(content)

def append_file(content, filename):
    # Appends the given content as a text file to the local directory
    with open(f"{filename}.txt", "a") as f:
        f.write(content)

def extract_msg_file(file_path) -> str:
    # Returns the contents of the msg in a text format
    LOGGER.info("Extracting msg")
    try:
        msg = extract_msg.Message(file_path)
    except Exception as e:
        LOGGER.error(f"Failed to extract msg from {file_path}: {e}")
        return ""

    formatted_date = msg.date.strftime("%A, %B %d, %Y %I:%M %p")

    msg_to_text =f"""
From: {msg.sender}
Sent: {formatted_date} 
To: {msg.to}
Cc: {msg.cc}
Subject: {msg.subject}
{msg.body} 
"""
    return msg_to_text
