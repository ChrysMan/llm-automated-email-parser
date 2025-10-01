import spacy, json, faiss, re
import numpy as np
from utils.logging_config import LOGGER
from utils.graph_utils import read_json_file, write_file


if __name__ == "__main__":
    try:
        json_sentences = read_json_file('/home/chryssida/DATA_TUC-KRITI/AIR EXPORT/230009/230009_qwen14.json')
        json_sentences.reverse()

        email_texts = [f"from: {item.get('from','')}\nsent: {item.get('sent','')}\nto:{item.get('to','')}\ncc:{item.get('cc','')}\nsubject:{item.get('subject','')}\nbody:{item.get('body','')}" 
                for item in json_sentences]
        email_bodies = [f"body:{item.get('body','')}" for item in json_sentences]
    except Exception as e:
        LOGGER.error(f"Failed to read JSON file: {e}")

    nlp = spacy.load("en_core_web_lg")

    unique_emails = []

    partially_unique_emails = list(dict.fromkeys(email_texts))
    expected_unique_emails_bodies = list(dict.fromkeys(email_bodies))
    print("\nLength of email before set", len(email_texts))
    print("\nLength of emails after set", len(partially_unique_emails))
    print("\nLength of email bodies before set", len(email_bodies))
    print("\nLength of emails bodies after set", len(expected_unique_emails_bodies))

    partially_unique_emails_bodies = [text.split("body:", 1)[1] for text in partially_unique_emails]

    if partially_unique_emails_bodies:
        doc1 = nlp(partially_unique_emails_bodies[0])
        embedding1 = np.array(doc1.vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(embedding1)

        dim = doc1.vector.shape[0]

        index = faiss.IndexFlatIP(dim)
        index.add(embedding1)
        unique_emails.append(email_texts[0])

        for body, email in zip(partially_unique_emails_bodies[1:], partially_unique_emails[1:]):
            doc = nlp(body)
            embedding = np.array(doc.vector, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(embedding)

            Dist, Ind = index.search(embedding, k=1)
            similarity = Dist[0][0]
            matched_index = Ind[0][0]
            mathed_email = unique_emails[matched_index]

            if similarity < 0.999:
                index.add(embedding)
                unique_emails.append(email)
            else:
                print(f"\n\nOriginal email: \n{email}\n\nMatched email:\n{mathed_email}\n\nSimilarity:\n{similarity}")
    else:
        LOGGER.error("No emails found to process.") 

    emails_json = []

for text in unique_emails:
    # Split at the first "body:" (case-insensitive, multi-line safe)
    parts = re.split(r'(?mi)^\s*body\s*:\s*', text, maxsplit=1)
    headers_part = parts[0]
    body_part = parts[1] if len(parts) == 2 else ""

    email_dict = {}
    for line in headers_part.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)  # split only on first colon
            email_dict[key.strip().lower()] = val.strip()

    email_dict["body"] = body_part.rstrip()
    emails_json.append(email_dict)

    # save to file
    with open('/home/chryssida/DATA_TUC-KRITI/AIR EXPORT/230009/230009_unique.json', "w", encoding="utf-8") as file:
        json.dump(emails_json, file, indent=4, ensure_ascii=False, default=str)