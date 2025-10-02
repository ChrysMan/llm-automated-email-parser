import spacy, json, faiss, re, os
import numpy as np
from utils.logging_config import LOGGER
from utils.graph_utils import read_json_file, write_file

FAISS_DB_PATH = "/home/chryssida/src/faiss_db.index"

if __name__ == "__main__":
    try:
        json_sentences = read_json_file('/home/chryssida/DATA_TUC-KRITI/SEA IMPORT/234107/234107_qwen14.json')
        json_sentences.reverse()

        email_texts = [f"from: {item.get('from','')}\nsent: {item.get('sent','')}\nto:{item.get('to','')}\ncc:{item.get('cc','')}\nsubject:{item.get('subject','')}\nbody:{item.get('body','')}" 
                for item in json_sentences]
        email_bodies = [f"body:{item.get('body','')}" for item in json_sentences]
    except Exception as e:
        LOGGER.error(f"Failed to read JSON file: {e}")

    nlp = spacy.load("en_core_web_lg")

    partially_unique_emails = list(dict.fromkeys(email_texts))
    expected_unique_emails_bodies = list(dict.fromkeys(email_bodies))
    print("\nLength of email before set", len(email_texts))
    print("\nLength of emails after set", len(partially_unique_emails))
    print("\nLength of email bodies before set", len(email_bodies))
    print("\nLength of emails bodies after set", len(expected_unique_emails_bodies))

    partially_unique_emails_bodies = [text.split("body:", 1)[1] for text in partially_unique_emails]

    
    dim = nlp(partially_unique_emails_bodies[0]).vector.shape[0]

    persistent_index = None
    dedup_index = None
    unique_emails = []
    emails_json = []

    if os.path.exists(FAISS_DB_PATH):
        print("Loading existing FAISS index...")
        persistent_index = faiss.read_index(FAISS_DB_PATH)
    else:
        print("Creating new FAISS index...")
        persistent_index = faiss.IndexFlatIP(dim)  # Using Inner Product (dot product) for similarity search


    for body, email in zip(partially_unique_emails_bodies, partially_unique_emails):
        """Embeddings for deduplication DB"""
        doc_bodies = nlp(body)
        embedding_bodies = np.array(doc_bodies.vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(embedding_bodies)

        if dedup_index is None:
            dim = doc_email.vector.shape[0]
            index_dedup = faiss.IndexFlatIP(dim)
            index_dedup.add(embedding_bodies)
            unique_emails.append(body)
            persistent_index.add(embedding_email) # add to persistent db

        else:
            Dist, Ind = index_dedup.search(embedding_bodies, k=1)
            similarity = Dist[0][0]
            matched_index = Ind[0][0]
            mathed_email = unique_emails[matched_index]

            if similarity < 0.999:
                index_dedup.add(embedding_bodies)   # add to in memory db

                """Embeddings for persistent DB"""
                doc_email = nlp(email)
                embedding_email = np.array(doc_email.vector, dtype=np.float32).reshape(1, -1)
                faiss.normalize_L2(embedding_email)
                unique_emails.append(email)
                persistent_index.add(embedding_email) # add to persistent db
            else:
                print(f"\n\nOriginal email: \n{email}\n\nMatched email:\n{mathed_email}\n\nSimilarity:\n{similarity}")
        
        faiss.write_index(persistent_index, FAISS_DB_PATH) # save persistent db
                

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
    with open('/home/chryssida/src/unique_emails.json', "a", encoding="utf-8") as file:
        json.dump(emails_json, file, indent=4, ensure_ascii=False, default=str)

