
from dataclasses import dataclass
from typing import Optional, Any
from langchain_openai import ChatOpenAI
from lightrag import LightRAG
from pydantic import SkipValidation

# @dataclass
# class AgentDeps:
#     """Dependencies for the agents."""
#     lightrag: LightRAG
#     refinement_llm: Optional[ChatOpenAI] = None

@dataclass
class AgentDeps:
    """Dependencies for the agents."""
    lightrag: SkipValidation[Any] 
    refinement_llm: Optional[SkipValidation[Any]] = None