import extract_msg, re
from .logging import LOGGER

def extract_msg_file(file_path) -> str:
    """ Returns the contents of the msg in a text format """
    LOGGER.info("Extracting msg...")
    try:
        msg = extract_msg.Message(file_path)
    except Exception as e:
        LOGGER.error(f"Failed to extract msg from {file_path}: {e}")
        return ""

    formatted_date = msg.date.strftime("%A, %B %d, %Y %I:%M %p")

    if msg.cc is None:
        msg.cc = ""
        
    msg_to_text =f"""From: {msg.sender}
Sent: {formatted_date} 
To: {msg.to}
Cc: {msg.cc}
Subject: {msg.subject}
{msg.body} 
"""
    return msg_to_text

def clean_data(text: str) -> str:
    """ Cleans the text data by removing unnecessary spaces, new lines and unecessary data. """
    flags = re.MULTILINE | re.IGNORECASE
    clean_text = text.replace("--", "").replace('"\'', '"').replace('\'"', '"').replace("：", ":").replace("！", "!").replace("，", ",")
    clean_text = re.sub(r"<image\d+\.(jpg|png)>|re: *|回复: *|Σχετ.: *|__+|FW:\s*|Fwd:\s*", "", clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r"^[ \t]+", "", clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r"^发件人:", "From: ", clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r"^发送日期:|^发送时间:", "Sent: ", clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r"^收件人:", "To: ", clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r"^抄送人:|^抄送:", "Cc: ", clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r"^主题:", "Subject: ", clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r"^Date:", "Sent:", clean_text, flags=re.MULTILINE) 

# Tel\s*:.*|
# (T|M)\s*:.*|
# E(-)?mail\s*:.*|
# Skype\s*:.*|
# Dir\s*:.*|
# Website\s*:.*|
# Web\s*:.*|
# Address\s*:.*|
# Fax\s*:.*|
# mob\..*|
# Mobile\s*:.*|
# Office\s*:.*|
# Phone\s*:.*|
# P\.*s\.*\s*:.*$|
# https?://\S+|<https?://\S+>|                  
# www\.\S+\s+<https?://\S+>|                     
# www\.[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}|
    pattern = r"""\s*<mailto:[^>]+>|
(T|M)\s*:.*|
(T|M)\s\+.*|
Skype\s*:.*|
P\.*s\.*\s*:.*$|
https?://\S+|<https?://\S+>|   
Disclaimer\s*:.*|
Στάλθηκε από το Ταχυδρομείο.*|
Sent from my.*|
地址\s*:.*|
分公司\s*:.*|
This message is sent from my mob* device"""
    clean_text = re.sub(pattern , "", clean_text, flags=flags)
    clean_text = re.sub(r"\n\s*\n*", "\n", clean_text)
    return clean_text

def split_email_thread(clean_text: str) -> list: 
    """ Separates the emails using the word "From" or "On...wrote:" as an indicator to separate (English, Greek, Israeli, Chinese, Russian, German, French, Italian, Spanish, Portuguese). """
    separator = re.compile(r"^(From:|Από:|מאת:|发件人:|De:|Von:|Da:|De:|От:|On .+ wrote:|Στις .+ έγραψε:|在 .+ 写道:|Le .+ a écrit:|Am .+ schrieb:|El .+ escribió:|Il .+ ha scritto:|Em .+ escreveu:|В .+ написал(а)?:)", re.MULTILINE)   
    
    # Find all occurrences of reply headers
    matches = list(separator.finditer(clean_text))

    # If no matches, return the whole email as one part
    if not matches:
        return [clean_text.strip()]

    # Store the email parts
    email_parts = []
    prev_end = 0

    for match in matches:
        start = match.start()

        # Get the section from the previous end up to this header
        part = clean_text[prev_end:start].strip()
        if part:
            email_parts.append(part)

        prev_end = start  # Next part starts from this header

    # Add the last section of the email (from the last header to the end)
    last_part = clean_text[prev_end:].strip()
    if last_part:
        email_parts.append(last_part)

    return email_parts