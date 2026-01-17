
from dataclasses import dataclass
from typing import Optional
from langchain_openai import ChatOpenAI
from lightrag import LightRAG

@dataclass
class AgentDeps:
    """Dependencies for the agents."""
    lightrag: LightRAG
    refinement_llm: Optional[ChatOpenAI] = None
    dir_path: Optional[str] = None

