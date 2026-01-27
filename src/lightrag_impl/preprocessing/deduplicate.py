import sys
import json, faiss, re, os
import numpy as np
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings

#from lightrag_impl.core.llm import hf
from utils.logging import LOGGER
from utils.file_io import read_json_file


def deduplicate_emails(dict_list: List[dict]) -> list[str]:
    try:
        embedder = HuggingFaceEmbeddings(
            model_name = "BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs = {'normalize_embeddings': True}  
        )
    except Exception as e:
        LOGGER.error(f"Error while initializing the embedder: {e}")
        return f"Error while initializing the embedder: {e}"
    try:
        dict_list.reverse()

        email_texts = [f"from: {item.get('from','')}\nsent: {item.get('sent','')}\nto:{item.get('to','')}\ncc:{item.get('cc','')}\nsubject:{item.get('subject','')}\nbody:{item.get('body','')}" 
                for item in dict_list]
    except Exception as e:
        LOGGER.error(f"Failed to read JSON file: {e}")
        return f"Failed to read JSON file: {e}"


    partially_unique_emails = list(dict.fromkeys(email_texts))

    partially_unique_emails_bodies = [text.split("body:", 1)[1] for text in partially_unique_emails]
    
    embeddings = embedder.embed_documents(partially_unique_emails_bodies)

    dedup_index = None
    unique_emails = []

    for body, email in zip(embeddings, partially_unique_emails):
        """Embeddings for deduplication DB"""

        body = np.array(body, dtype=np.float32).reshape(1, -1)

        
        if dedup_index is None:
            try:
                dim = len(body[0]) 
                #dim = doc_bodies.vector.shape[0]
                dedup_index = faiss.IndexFlatIP(dim)
                dedup_index.add(body)
                unique_emails.append(email)
            except Exception as e:
                LOGGER.error(f"Error here: {e}")

        else:
            try:
                Dist, Ind = dedup_index.search(body, k=1)
                similarity = Dist[0][0]
                matched_index = Ind[0][0]
                mathed_email = unique_emails[matched_index]
            except Exception as e:
                LOGGER.error(f"Error here!: {e}")

            if similarity < 0.983:
                dedup_index.add(body)   # add to in memory db
                unique_emails.append(email)
            else:
                #continue
                print(f"\n\nOriginal email: \n{email}\n\nMatched email:\n{mathed_email}\n\nSimilarity:\n{similarity}")
    return unique_emails

if __name__ == "__main__":
    if len(sys.argv) != 2:
        LOGGER.error("Usage: python deduplicate.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    #file_path = "/home/chryssida/DATA_TUC-KRITI/AIR EXPORT/230009/230009.json"

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
    