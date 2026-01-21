import faiss, os, spacy, sys
import numpy as np
from typing import Any
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from preprocessing_implementations.vllm_serve import LLMPredictor
from lightrag_impl.prompts.preprocessing_prompts import rag_prompt
from utils.file_io import read_json_file

from utils.logging import LOGGER

FAISS_DB_PATH = "/home/chryssida/llm-automated-email-parser/src/faiss_db.index"
EMBED_MODEL = "BAAI/bge-m3"

def init_embed_hf(model: str)->HuggingFaceEmbeddings:
    hf = HuggingFaceEmbeddings(
        model_name = model,
        model_kwargs={"device": "cpu"},
        encode_kwargs = {'normalize_embeddings': True}  
    )
    return hf

def init_embed_spacy()->SpacyEmbeddings:
    nlp = spacy.load("en_core_web_lg")
    #embeddings = np.array([doc.vector for doc in map(nlp, unique_emails)], dtype=np.float32)
    embedder = SpacyEmbeddings(model_name="en_core_web_lg", nlp=nlp)
    return embedder

def index_faiss_db(unique_emails: list[str], embedder: HuggingFaceEmbeddings|SpacyEmbeddings):

    embeddings = embedder.embed_documents(unique_emails)
    #faiss.normalize_L2(embeddings)

    if os.path.exists(FAISS_DB_PATH): # Check if they are added twice
        try:
            LOGGER.info("Loading existing FAISS index...")
            vectorstore = FAISS.load_local(FAISS_DB_PATH, embedder, allow_dangerous_deserialization=True)
            vectorstore.add_embeddings(zip(unique_emails, embeddings))
            vectorstore.save_local(FAISS_DB_PATH)
        except Exception as e:
            LOGGER.error(f"Error while loading the vectorstore: {e}")
            sys.exit(1)
            
    else:
        try:
            LOGGER.info("Creating new FAISS index")
            index = faiss.IndexFlatIP(len(embeddings[0]))
            vectorstore = FAISS(
                embedding_function=embedder,
                index=index,
                docstore= InMemoryDocstore(),
                normalize_L2=True,
                index_to_docstore_id={}
                )
            vectorstore.add_embeddings(zip(unique_emails, embeddings))
            vectorstore.save_local(FAISS_DB_PATH)
        except Exception as e:
            LOGGER.error(f"Error while creating the vectorstore: {e}")
            sys.exit(1)

def retrieve_with_normalized_scores(embedder: HuggingFaceEmbeddings|SpacyEmbeddings, vectorstore:FAISS, query: str, k: int = 5, score_threshold: float = 0.0):
    """
    Returns top-k documents with cosine similarity.
    Only returns results above score_threshold.
    """
    # ------- Using hf embedder -------
    query_embedding = embedder.embed_query(query)
    # ---------------------------------

    # ------- Using spacy embedder -------
    # query_embedding = embedder.embed_query(query)
    # query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    # faiss.normalize_L2(query_embedding)
    # ------------------------------------

    results = vectorstore.similarity_search_with_score_by_vector(query_embedding, k=k, kwargs = {"score_threshold": score_threshold})

    return results

if __name__ == "__main__":
    #file_path = "/home/chryssida/DATA_TUC-KRITI/AIR EXPORT/230009/230009.json"
    file_path= ""
    predictor = LLMPredictor()
    embedder = init_embed_hf(EMBED_MODEL)
    if os.path.exists(file_path):
        try:
            
            list_of_dicts = read_json_file(file_path)
            list_of_strs = list(map(str, list_of_dicts))

            index_faiss_db(list_of_strs, embedder)

            vectorstore = FAISS.load_local(FAISS_DB_PATH, embedder, allow_dangerous_deserialization=True)
        except Exception as e:
            LOGGER.error(f"Error while creating or updating vector database: {e}")
    else:
        vectorstore = FAISS.load_local(FAISS_DB_PATH, embedder, allow_dangerous_deserialization=True)
            
    while True:
        query = input("\nEnter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        # docs = retriever.invoke(query)
        # emails = [doc.page_content for doc in docs]

        contexts = retrieve_with_normalized_scores(embedder, vectorstore, query, k=15, score_threshold=0.4)
        ctx_str = "\n\n".join([doc.page_content for doc, _ in contexts])

        prompt = rag_prompt.format(context_data=ctx_str, query=query)

        result = predictor.process_single_prompt(prompt)

        print("\nTop similar emails:")
        for i ,(doc, score) in enumerate(contexts, 1):
            print(f"\nEmail {i}, Score {score:.4f}:\n{doc.page_content}\n")

        print(f"\n\nResponse:\n{result}")
        