import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from lightrag_implementation.basic_operations import initialize_rag
from lightrag_implementation.agents.agent_deps import AgentDeps
from lightrag_implementation.agents.supervisor import create_supervisor_agent
from utils.graph_utils import find_dir

load_dotenv()

WORKING_DIR = find_dir("rag_storage", "./")
os.makedirs(WORKING_DIR, exist_ok=True)

app = FastAPI()

class ChatInput(BaseModel):
    message: str
    message_history: list | None = None

class ChatOutput(BaseModel):
    response: str
    all_messages: list

@app.on_event("startup")
async def startup_event():
    rag = await initialize_rag(working_dir=WORKING_DIR)

    ref_llm = ChatOpenAI(
        temperature=0.2,
        model=os.getenv("LLM_MODEL"),
        base_url=os.getenv("LLM_BINDING_HOST"),
        api_key=os.getenv("LLM_BINDING_API_KEY"),
    )

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

        return ChatOutput(
            response=result.output,
            all_messages=result.all_messages(),
        )

    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

@app.get("/health")
async def health():
    rag = getattr(app.state, "rag", None)
    session_alive = getattr(rag, "session", None) is not None
    return {"rag_session_alive": session_alive}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
