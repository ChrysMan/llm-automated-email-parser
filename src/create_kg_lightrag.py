import os
from utils.logging_config import LOGGER
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc
import pdfplumber

rag = LightRAG