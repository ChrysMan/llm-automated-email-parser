import spacy
import faiss
import numpy as np
from utils.logging_config import LOGGER
from utils.graph_utils import read_json_file, write_file


try:
    json_sentences = read_json_file('/home/chryssida/DATA_TUC-KRITI/SEA IMPORT/234107/234107.json')
    
    email_texts = [f"sender: {item.get('sender','')}\nsent:{item.get('sent','')}\nto:{item.get('to','')}\ncc:{item.get('cc','')}\nsubject:{item.get('subject','')}\nbody:{item.get('body','')}" 
              for item in json_sentences]
except Exception as e:
    LOGGER.error(f"Failed to read JSON file: {e}")

nlp = spacy.load("en_core_web_lg")

unique_emails = []

if email_texts:
    doc1 = nlp(email_texts[0])
    embedding1 = np.array(doc1.vector, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(embedding1)

    dim = doc1.vector.shape[0]

    index = faiss.IndexFlatIP(dim)
    index.add(embedding1)
    unique_emails.append(email_texts[0])

    for email in email_texts[1:]:
        doc = nlp(email)
        embedding = np.array(doc.vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(embedding)

        Dist, Ind = index.search(embedding, k=1)
        similarity = Dist[0][0]
        matched_index = Ind[0][0]
        mathed_email = unique_emails[matched_index]

        if similarity < 0.96:
            index.add(embedding)
            unique_emails.append(email)
        else:
            print(f"\n\nOriginal email: \n{email}\n\nMatched email:\n{mathed_email}\n\nSimilarity:\n{similarity}")
else:
    LOGGER.error("No emails found to process.") 

write_file(unique_emails, '/home/chryssida/DATA_TUC-KRITI/SEA IMPORT/234107/234107_unique.txt')
