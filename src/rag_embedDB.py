import spacy, faiss, os
from utils.logging_config import LOGGER
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings

FAISS_DB_PATH = "/home/chryssida/src/faiss_db.index"

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

        docs = retriever.invoke(query)
        emails = [doc.page_content for doc in docs]

        print("\nTop similar emails:")
        for i in range(len(emails)):
            print(f"\nEmail {i+1}:\n{emails[i]}\n")
        