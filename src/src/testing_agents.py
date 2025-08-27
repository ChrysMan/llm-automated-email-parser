import os
import torch
import re
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
if langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
    os.environ["LANGSMITH_PROJECT"] = "email_preprocessing"
else:
    LOGGER.warning("Langsmith API key not found. Tracing will be disabled.")

from langsmith import traceable, trace, Client
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, AutoModelForSeq2SeqLM
from utils.logging_config import LOGGER
from utils.prompts import translator_prompt, translator_prompt_qwen2, translator_prompt_Llama, headers_cleaning_prompt, signature_cleaning_prompt
from agents.preprocessing_agent import clean_email_llm, extract_email_llm
from agents.translator_agent import translate_email_llm


if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model_name2 = "LuvU4ever/qwen2.5-3b-qlora-merged-v4"
    #model_name2 = "Helsinki-NLP/opus-mt-mul-en"
    #model_name2 = "LLaMAX/LLaMAX2-7B-Alpaca"
    #model_name2 = "winninghealth/WiNGPT-Babel-2-AWQ"

    device0 = "cuda:0" if torch.cuda.is_available() else "cpu"
    device1 = "cuda:1" if torch.cuda.is_available() else "cpu"
    
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype="float16",
    #     attn_implementation="sdpa",
    #     device_map="auto"
    # )#.to(device0)

    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    model2 = AutoModelForCausalLM.from_pretrained(
        model_name2,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    ).to(device1)

    tokenizer2 = AutoTokenizer.from_pretrained(model_name2)

    email_text = """your email goes here"""

    prompt = """Translate this to English."""
    translated_email = translate_email_llm(email_text, prompt=prompt, model=model2, tokenizer=tokenizer2, trace_name="translate_format_email", device=device1)
    print("\n\nTranslated output: \n", translated_email)

    # cleaned_from_signatures = clean_email_llm(formatted_email, prompt=signature_cleaning_prompt, model=model, tokenizer=tokenizer, trace_name="clean_email_signatures", device=device0)
    # print("\n\nCleaned signatures: \n", cleaned_from_signatures)
    # cleaned_from_headers = clean_email_llm(cleaned_from_signatures, prompt=headers_cleaning_prompt, model=model, tokenizer=tokenizer, trace_name="clean_email_headers", device=device0)
    # print("\n\nCleaned headers: \n", cleaned_from_headers)

    # extracted_info = extract_email_llm(email_text, prompt=extraction_prompt, model=model, tokenizer=tokenizer, trace_name="email1", device=device)
    # print(f"Extracted Information:\n", json.dumps(extracted_info, indent=2, ensure_ascii=False, default=str))
    