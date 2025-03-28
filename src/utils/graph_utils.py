import extract_msg, re
from utils.logging_config import LOGGER

def write_file(content, filename):
    """ Writes the given content to the given file to the local directory """
    try:
        with open(f"{filename}", "w") as f:
            f.write(str(content))
    except Exception as e:
        LOGGER.error(f"Failed to write at file {filename}: {e}")

def append_file(content, filename):
    """ Appends the given content to given file to the local directory """
    try:
        with open(f"{filename}", "a") as f:
            f.write(str(content))
    except Exception as e:
        LOGGER.error(f"Failed to append at file {filename}: {e}")

def extract_msg_file(file_path) -> str:
    """ Returns the contents of the msg in a text format """
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

def split_email_chain(text): 
    """ Separates the emails using the word "From" or "On...wrote:" as an indicator to seperate. """
    seperator = re.compile(r"(?<=\n)\s*(From:|On .+ wrote:)", re.MULTILINE)
    email_parts = re.split(seperator, text)
    email_parts = [part.strip() for part in email_parts if part.strip()]
    formatted_parts = []
    for i, part in enumerate(email_parts, 1):
        if i % 2 ==1:
            formatted_parts.append(f"{part}")
        else:
            clean_part = re.sub(r"^\s+|\s{2,}", "\n", part)
            formatted_parts[-1] += f"{clean_part}"
    return formatted_parts
