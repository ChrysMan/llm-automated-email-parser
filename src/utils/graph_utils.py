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
    LOGGER.info("Extracting msg...")
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

def clean_data(text: str) -> str:
    """ Cleans the text data by removing unnecessary spaces and new lines. """
    clean_text = re.sub(r"^[ \t]+", "", text, flags=re.MULTILINE)
    clean_text = re.sub(r"\n\s*\n+", "\n\n", clean_text)
    return clean_text

def split_email_thread(text: str) -> list: 
    """ Separates the emails using the word "From" or "On...wrote:" as an indicator to seperate. """
    seperator = re.compile(r"(?<=\n)\s*(From:|On .+ wrote:)", re.MULTILINE)
    email_parts = re.split(seperator, text)
    email_parts = [part.strip() for part in email_parts if part.strip()]
    formatted_parts = []
    for i, part in enumerate(email_parts, 1):
        if i % 2 ==1:
            formatted_parts.append(f"{part}")
        else:
            part = part.replace("________________________________", "")
            part = part.replace("--", "")
            part = part.replace('"\'', '"').replace('\'"', '"')
            formatted_parts[-1] += f"{part}"
    return formatted_parts

def chunk_emails(email_list, chunk_size=10):
    """ Yield successive chunks of n emails from the list. """
    for i in range(0, len(email_list), chunk_size):
        yield email_list[i:i + chunk_size]
