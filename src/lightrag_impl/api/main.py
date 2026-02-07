import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Optional, Literal

from ..core.llm import ref_llm
from ..core.pipeline import initialize_rag, run_async_query
from ..agents.dependencies import AgentDeps
from ..agents.supervisor import create_supervisor_agent
from utils.file_io import find_dir

load_dotenv()

WORKING_DIR = find_dir("rag_storage", "./")
os.makedirs(WORKING_DIR, exist_ok=True)

app = FastAPI()

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatInput(BaseModel):
    message: str
    message_history: Optional[List[ChatMessage]] = None

class ChatOutput(BaseModel):
    response: str

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: dict
    retrieved_contexts: Optional[dict] = None

@app.on_event("startup")
async def startup_event():
    rag = await initialize_rag(working_dir=WORKING_DIR)

    app.state.rag = rag

    app.state.deps = AgentDeps(
        lightrag=rag,
        refinement_llm=ref_llm,
        dir_path=None
    )

    app.state.supervisor_agent = create_supervisor_agent()

@app.post("/chat", response_model=ChatOutput)
async def chat_endpoint(chat_input: ChatInput):
    deps = getattr(app.state, "deps", None)
    supervisor_agent = getattr(app.state, "supervisor_agent", None)
    
    if deps is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        result = await supervisor_agent.run(
            chat_input.message,
            deps=deps,
            message_history=chat_input.message_history,
        )

        return ChatOutput(response=result.output)

    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

@app.post("/query")
async def simple_query_endpoint(request_data: QueryRequest):
    query_text = request_data.query

    deps = getattr(app.state, "deps", None)
    supervisor_agent = getattr(app.state, "supervisor_agent", None)
    
    if deps is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        result = await run_async_query(
            rag=deps.lightrag,
            question=query_text,
            mode="mix"
        )
        
        return QueryResponse(
            answer=result.get("llm_response", ""),
            retrieved_contexts=result.get("data", {})
        )

    except Exception as ex:
        print(f"!!! RAG ERROR: {type(ex).__name__}: {str(ex)}")
        raise HTTPException(status_code=500, detail=str(ex))

@app.get("/health")
async def health():
    rag = getattr(app.state, "rag", None)
    session_alive = getattr(rag, "session", None) is not None
    return {"rag_session_alive": session_alive}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
