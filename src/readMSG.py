import os
import extract_msg, re

directory = "/home/chryssida/DATA_TUC-KRITI/SEA EXPORT/230054/"
output_file = "/home/chryssida/src/Texts/SE-230054-info.txt"
output_file2 = "/home/chryssida/src/Texts/SE-230054-cleaned.txt"
output_file3 = "/home/chryssida/src/Texts/SE-230054-split.txt"


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
                msg = extract_msg.Message(file_path)

                # Format the datetime object back into a string
                formatted_date = msg.date.strftime("%A, %B %d, %Y %I:%M %p")

                if msg.cc is None:
                    msg.cc = ""
                '''file.write("-"*20 + f"{filename}"+ "-"*20 + "\n\n")
                file.write(f"From: {msg.sender}\n")
                file.write(f"Sent: {formatted_date}\n")
                file.write(f"To: {msg.to}\n")
                file.write(f"Cc: {msg.cc}\n")
                file.write(f"Subject: {msg.subject}\n")
                
                file.write(f"\n{msg.body}\n\n")'''
                msg_to_text +=f"""From: {msg.sender}
Sent: {formatted_date} 
To: {msg.to}
Cc: {msg.cc}
Subject: {msg.subject}
{msg.body}
"""           
                
        f.write(msg_to_text)

        print(f"Email information has been exported to {output_file}")
except Exception as e:
    print(f"An error occurred while writing to the file: {e}")


try:
    with open(f"{output_file2}", "w") as f:
        """ Cleans the text data by removing unnecessary spaces and new lines. """
        flags = re.MULTILINE | re.IGNORECASE
        clean_text = msg_to_text.replace("--", "").replace('"\'', '"').replace('\'"', '"').replace("：", ":")
        clean_text = re.sub(r"<image\d+\.(jpg|png)>|re: *|回复: *|Σχετ.: *|__+", "", clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r"^[ \t]+", "", clean_text, flags=re.MULTILINE) 
        clean_text = re.sub(r"Tel\s*:\s*.+$|^(T|M)\s*:\s*\+*.+$|E(-)?mail\s*:.*$|Website\s*:.*$|Web\s*: .+$|Address\s*:.+$|Fax\s*:.+$|P\.*s\.*\s*:.+$|mob\.\+*.+\s*$|Mobile\s*:.+$|Note\s*:.*$|Phone\s*:\s*.*$|Disclaimer\s*:.+$|Στάλθηκε από το Ταχυδρομείο.+$|Sent from my.+$|地址\s*:.+$|分公司\s*:.+$", "", clean_text, flags=flags)
        clean_text = re.sub(r"\n\s*\n*", "\n", clean_text)
        
        f.write(clean_text)
        print(f"Cleaned email information has been exported to {output_file2}")
except Exception as e:
    print(f"An error occurred while writing to the file: {e}")

try:
    with open(f"{output_file3}", "a") as f:
        separator = re.compile(r"^(From:|发件人:|De:|Von:|Da:|De:|От:|On .+ wrote:|Στις .+ έγραψε:|在 .+ 写道:|Le .+ a écrit:|Am .+ schrieb:|El .+ escribió:|Il .+ ha scritto:|Em .+ escreveu:|В .+ написал(а)?:)", re.MULTILINE)
        email_parts = split_email_chain(clean_text)

        # Join them with your separator
        joined = "\n-***-\n".join(email_parts)

        f.write(joined)
        print(f"Split email information has been exported to {output_file3}")
except Exception as e:
    print(f"An error occurred while writing to the file: {e}")








