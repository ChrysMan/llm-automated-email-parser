import spacy, json, faiss, re, os
import numpy as np
from utils.logging_config import LOGGER
from utils.graph_utils import read_json_file, write_file

FAISS_DB_PATH = "/home/chryssida/src/emails.index"

if __name__ == "__main__":
    if os.path.exists(FAISS_DB_PATH):
        index = faiss.read_index(FAISS_DB_PATH)
        print(f"Loaded FAISS index with {index.ntotal} vectors.")
    else:
        print("FAISS index not found. Please create the index first.")
        exit(1)

    nlp = spacy.load("en_core_web_lg")

    while True:
        query = input("\nEnter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        doc_query = nlp(query)
        embedding_query = np.array(doc_query.vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(embedding_query)

        k = 5  # number of nearest neighbors to retrieve
        Dist, Ind = index.search(embedding_query, k)

        retrieved_emails = [emails[i] for i in Ind[0]]

        print("\nTop similar emails:")
        for i in range(k):
            index = Ind[0][i]
            similarity = Dist[0][i]
            if index < len(unique_emails):
                print(f"\nEmail {i+1} (Similarity: {similarity:.4f}):\n{unique_emails[index]}")
            else:
                print(f"\nEmail {i+1} (Similarity: {similarity:.4f}):\nIndex out of range")

        