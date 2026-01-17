import faiss, os
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings

from lightrag_impl.utils.logging import LOGGER

FAISS_DB_PATH = "/home/chryssida/src/faiss_db.index"

def retrieve_with_normalized_scores(query: str, k: int = 5, score_threshold: float = 0.0):
    """
    Returns top-k documents with cosine similarity.
    Only returns results above score_threshold.
    """
    query_embedding = embedder.embed_query(query)
    query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    
    faiss.normalize_L2(query_embedding)
    
    results = vectorstore.similarity_search_with_score_by_vector(query_embedding[0], k=k)
    
    # Normalize similarity scores to [0,1]
    #normalized_results = [(doc, (score + 1) / 2) for doc, score in results]
    
    filtered_results = [(doc, score) for doc, score in results if score >= score_threshold]
    
    return filtered_results

if __name__ == "__main__":
    if os.path.exists(FAISS_DB_PATH):
        embedder = SpacyEmbeddings(model_name="en_core_web_lg")

        vectorstore = FAISS.load_local(FAISS_DB_PATH, embedder, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        
        print(f"Loaded FAISS vectorstore.")
    else:
        print("FAISS index not found. Please create the index first.")
        exit(1)
    
    while True:
        query = input("\nEnter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        #docs = retriever.invoke(query)
        #emails = [doc.page_content for doc in docs]

        results = retrieve_with_normalized_scores(query, k=6, score_threshold=0.4)

        print("\nTop similar emails:")
        for i ,(doc, score) in enumerate(results, 1):
            print(f"\nEmail {i}, Score {score:.4f}:\n{doc.page_content}\n")
        