
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

class LightRAGBox:
    """A simple container to hold the active LightRAG instance."""
    def __init__(self, instance):
        self.instance = instance

@dataclass
class AgentDeps:
    def __init__(self, rag_box: LightRAGBox, refinement_llm=None):
        self.rag_box = rag_box
        self.refinement_llm = refinement_llm

    @property
    def lightrag(self):
        return self.rag_box.instance
    
    @lightrag.setter
    def lightrag(self, new_instance):
        self.rag_box.instance = new_instance