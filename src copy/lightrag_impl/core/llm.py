
import os
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from langchain_openai import ChatOpenAI
from functools import partial
from lightrag.rerank import generic_rerank_api
from lightrag.llm.openai import openai_complete_if_cache


agent_llm = OpenAIChatModel(
    os.getenv("LLM_AGENT_MODEL", "Qwen/Qwen2.5-3B-Instruct"),
    provider = OpenAIProvider(
        base_url=os.getenv("LLM_AGENT_BINDING_HOST"), 
        api_key=os.getenv("LLM_AGENT_BINDING_API_KEY")
    )    
)

ref_llm = ChatOpenAI(
            temperature=0.2, 
            model=os.getenv("LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8"), 
            base_url=os.getenv("LLM_BINDING_HOST"), 
            api_key=os.getenv("LLM_BINDING_API_KEY")
        )

rerunk_func = partial(
    generic_rerank_api, 
    model=os.getenv("RERANK_MODEL"),
    base_url=os.getenv("RERANK_BINDING_HOST"),
    api_key=os.getenv("RERANK_BINDING_API_KEY")
)

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        os.getenv("LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BINDING_HOST"),
        **kwargs,
    )

