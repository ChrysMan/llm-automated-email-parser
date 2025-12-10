import sys
import spacy, json, faiss, re, os
import numpy as np
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from utils.logging_config import LOGGER
from utils.graph_utils import read_json_file, write_file

FAISS_DB_PATH = "/home/chryssida/src/faiss_db.index"

def deduplicate_emails(dict_list: List[dict]) -> list[str]:
    try:
        dict_list.reverse()

        email_texts = [f"from: {item.get('from','')}\nsent: {item.get('sent','')}\nto:{item.get('to','')}\ncc:{item.get('cc','')}\nsubject:{item.get('subject','')}\nbody:{item.get('body','')}" 
                for item in dict_list]
        email_bodies = [f"body:{item.get('body','')}" for item in dict_list]
    except Exception as e:
        LOGGER.error(f"Failed to read JSON file: {e}")

    nlp = spacy.load("en_core_web_lg")

    partially_unique_emails = list(dict.fromkeys(email_texts))
    expected_unique_emails_bodies = list(dict.fromkeys(email_bodies))
    # print("\nLength of email before set", len(email_texts))
    # print("\nLength of emails after set", len(partially_unique_emails))
    # print("\nLength of email bodies before set", len(email_bodies))
    # print("\nLength of emails bodies after set", len(expected_unique_emails_bodies))

    partially_unique_emails_bodies = [text.split("body:", 1)[1] for text in partially_unique_emails]
    
    dim = nlp(partially_unique_emails_bodies[0]).vector.shape[0]

    dedup_index = None
    unique_emails = []

    for body, email in zip(partially_unique_emails_bodies, partially_unique_emails):
        """Embeddings for deduplication DB"""
        doc_bodies = nlp(body)
        embedding_bodies = np.array(doc_bodies.vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(embedding_bodies)

        if dedup_index is None:
            dim = doc_bodies.vector.shape[0]
            dedup_index = faiss.IndexFlatIP(dim)
            dedup_index.add(embedding_bodies)
            unique_emails.append(email)

        else:
            Dist, Ind = dedup_index.search(embedding_bodies, k=1)
            similarity = Dist[0][0]
            matched_index = Ind[0][0]
            mathed_email = unique_emails[matched_index]

            if similarity < 0.992:
                dedup_index.add(embedding_bodies)   # add to in memory db
                unique_emails.append(email)
            else:
                continue
                #print(f"\n\nOriginal email: \n{email}\n\nMatched email:\n{mathed_email}\n\nSimilarity:\n{similarity}")
    return unique_emails

def create_faiss_db(unique_emails: list[str]):
    """Create/Update persistent FAISS DB from unique emails"""
    nlp = spacy.load("en_core_web_lg")
    embeddings = np.array([doc.vector for doc in map(nlp, unique_emails)], dtype=np.float32)
    faiss.normalize_L2(embeddings)


    if os.path.exists(FAISS_DB_PATH):
        print("Loading existing FAISS index...")
        vectorstore = FAISS.load_local(FAISS_DB_PATH, SpacyEmbeddings(model_name="en_core_web_lg"), allow_dangerous_deserialization=True)
        vectorstore.add_embeddings(zip(unique_emails, embeddings))
    else:
        print("Creating new FAISS index...")
        index = faiss.IndexFlatIP(embeddings.shape[1])  # Using Inner Product (dot product) for similarity search
        embedder = SpacyEmbeddings(model_name="en_core_web_lg")
        vectorstore = FAISS(
            embedding_function=embedder,
            index=index,
            docstore= InMemoryDocstore(),
            index_to_docstore_id={}
            )
        vectorstore.add_embeddings(zip(unique_emails, embeddings))
    vectorstore.save_local(FAISS_DB_PATH)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        LOGGER.error("Usage: python create_vectorDB_spacy_faiss.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.isfile(file_path):
        LOGGER.error(f"{file_path} is not a valid file.")
        sys.exit(1)

    dir_path = os.path.dirname(file_path)
    try:
        json_sentences = read_json_file(file_path)
        unique_emails = deduplicate_emails(json_sentences)

        output_path = os.path.join(dir_path, f"{os.path.basename(dir_path)}_unique.json")
        
        emails_json = []
        #write_file("", output_path)  

        print(len(unique_emails))
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
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(emails_json, file, indent=4, ensure_ascii=False, default=str)

        LOGGER.info(f"Unique emails saved to {output_path}")
    except Exception as e:
        LOGGER.error(f"Error during deduplication: {e}")
        sys.exit(1)
    
    #create_faiss_db(unique_emails)