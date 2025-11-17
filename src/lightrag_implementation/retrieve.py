from lightrag.lightrag import LightRAG, QueryParam

async def run_async_query(rag: LightRAG, question: str, mode: str, top_k: int = 5) -> str:
    """
    Execute an async RAG query using .aquery method
    """
    return await rag.aquery(
        query=question,
        param=QueryParam(mode=mode, enable_rerank=True) #top_k=top_k,
        #system_prompt="You are a helpful assistant that provides accurate and concise information. Answer the user's question based on the retrieved documents."
    )