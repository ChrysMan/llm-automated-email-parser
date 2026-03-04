import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Any

from graphrag_impl.agent import create_supervisor_agent
from utils.file_io import find_dir

load_dotenv()

WORKING_DIR = find_dir("rag_storage", "./")
os.makedirs(WORKING_DIR, exist_ok=True)

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: List[Any]

@app.on_event("startup")
async def startup_event():
    app.state.supervisor_agent = create_supervisor_agent()

@app.post("/query")
async def simple_query_endpoint(request_data: QueryRequest):
    query_text = request_data.query
    
    if getattr(app.state, "supervisor_agent", None) is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        result = app.state.supervisor_agent.invoke(
         {"messages": [{"role": "user", "content": query_text}]}
    )
        
        return QueryResponse(
            response=result['messages']
        )

    except Exception as ex:
        print(f"!!! RAG ERROR: {type(ex).__name__}: {str(ex)}")
        raise HTTPException(status_code=500, detail=str(ex))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
