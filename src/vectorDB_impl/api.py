from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from vectorDB_impl.rag_vectorDB import init_embed_hf, load_vectorstore, vectordb_retrieval

load_dotenv()

EMBED_MODEL = "BAAI/bge-m3" 

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str
    context: str

@app.on_event("startup")
async def startup_event():
    app.state.embedder = init_embed_hf(EMBED_MODEL)
    app.state.vectorstore = load_vectorstore(app.state.embedder)

@app.post("/query")
async def simple_query_endpoint(request_data: QueryRequest):
    query_text = request_data.query
    
    if getattr(app.state, "vectorstore", None) is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        result, ctx_str = vectordb_retrieval(
            query_text, app.state.embedder, app.state.vectorstore
        )
        
        return QueryResponse(
            response=result,
            context=ctx_str
        )

    except Exception as ex:
        print(f"!!! RAG ERROR: {type(ex).__name__}: {str(ex)}")
        raise HTTPException(status_code=500, detail=str(ex))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
