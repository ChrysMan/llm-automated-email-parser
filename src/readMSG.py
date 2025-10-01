import os
import extract_msg, re
from utils.graph_utils import extract_msg_file, clean_data, split_email_thread

directory = "/home/chryssida/DATA_TUC-KRITI/AIR EXPORT/230009/"
output_file = "/home/chryssida/src/Texts/AE-230009-info.txt"
output_file2 = "/home/chryssida/src/Texts/AE-230009-cleaned.txt"
output_file3 = "/home/chryssida/src/Texts/AE-230009-split.txt"


def split_email_chain(clean_text):
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

try:
    with open(f"{output_file3}", "w") as f:
        f.write("")
except Exception as e:
    print(f"Failed to write at file {output_file3}: {e}")

try:      
    with open(f"{output_file}", "w") as f:

        msg_to_text = ""

        for filename in os.listdir(directory):
            if filename.endswith(".msg"):
                file_path = os.path.join(directory,filename)
                msg_to_text += extract_msg_file(file_path)
                
        f.write(msg_to_text)

        print(f"Email information has been exported to {output_file}")
except Exception as e:
    print(f"An error occurred while writing to the file: {e}")


try:
    with open(f"{output_file2}", "w") as f:
        """ Cleans the text data by removing unnecessary spaces and new lines. """
        clean_text = clean_data(msg_to_text)

        f.write(clean_text)
        print(f"Cleaned email information has been exported to {output_file2}")
except Exception as e:
    print(f"An error occurred while writing to the file: {e}")

try:
    with open(f"{output_file3}", "a") as f:
        email_parts = split_email_thread(clean_text)

        # Join them with your separator
        joined = "\n-***-\n".join(email_parts)

        f.write(joined)
        print(f"Split email information has been exported to {output_file3}")
except Exception as e:
    print(f"An error occurred while writing to the file: {e}")








