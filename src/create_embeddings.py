import faiss
import numpy as np
from time import time
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from utils.graph_utils import read_json_file, write_file
from utils.logging_config import LOGGER

tic = time()

model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-4B",
    model_kwargs={"attn_implementation": "sdpa", "torch_dtype": "torch.float16"},# "device_map": "auto"},
    tokenizer_kwargs={"padding_side": "left"}
)

try:
    json_sentences = read_json_file('/home/chryssida/DATA_TUC-KRITI/SEA IMPORT/231870/231870_vllm.json')
    
    email_texts = [f"sender: {item.get('sender','')}\nsent:{item.get('sent','')}\nto:{item.get('to','')}\ncc:{item.get('cc','')}\nsubject:{item.get('subject','')}\nbody:{item.get('body','')}" 
              for item in json_sentences]
except Exception as e:
    LOGGER.error(f"Failed to read JSON file: {e}")

if email_texts:
    #print(f"Number of emails: {len(email_texts)}\n")
    #print("First email text:", email_texts[0],"\n")
    partially_unique_emails = list(set(email_texts))  # Partially Remove exact duplicates
    #print("First partially email text:", partially_unique_emails[0],"\n")
    #print("\n", len(partially_unique_emails), " partially unique emails found.")
    unique_emails = []
else:
    LOGGER.error("No email texts found in the JSON file.")
    partially_unique_emails = []
    unique_emails = []

if partially_unique_emails:
    first_embedding = model.encode(partially_unique_emails[0], convert_to_numpy=True)
    first_embedding = np.array(first_embedding).reshape(1,-1)  # Reshape to 2D array

    print(f"type: {type(first_embedding)}")
    print(f"dtype: {getattr(first_embedding, 'dtype', None)}")
    print(f"shape: {getattr(first_embedding, 'shape', None)}")
    
    vector_dimensions = first_embedding.shape[1]  

    index = faiss.IndexFlatIP(vector_dimensions)  # Inner Product (dot product) index
    faiss.normalize_L2(first_embedding)           # Normalize embeddings
    index.add(first_embedding)                    # Add embeddings to the index
    unique_emails.append(partially_unique_emails[0])          # Add the first email to the unique list

    for email in partially_unique_emails[1:]:
        email_embedding = model.encode(email, convert_to_numpy=True)
        email_embedding = np.array(email_embedding).reshape(1,-1)  # Reshape to 2D array
        faiss.normalize_L2(email_embedding) 

        Dist, Ind = index.search(email_embedding, k=1)   # Search for the nearest neighbor
        similarity = Dist[0][0]                          # Get the similarity score
        matched_index = Ind[0][0]                        # Get the index of the matched embedding
        matched_email = partially_unique_emails[matched_index]        # Get the matched email text
        
        print("Similarity:", similarity, "\nOriginal Email:", email, "\nMatched Email:", matched_email)

        if similarity < 0.9:                             # If the similarity is below a threshold
            index.add(email_embedding)                   # Add the new embedding to the index
            unique_emails.append(email)

write_file(unique_emails, '/home/chryssida/DATA_TUC-KRITI/SEA IMPORT/231870/231870_vllm_unique.txt')
write_file(partially_unique_emails, '/home/chryssida/DATA_TUC-KRITI/SEA IMPORT/231870/231870_vllm_part_unique.txt')

print(f"Time taken to process: {time() - tic} seconds")
