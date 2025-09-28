import os, json
import torch
import re
from email import message_from_string
from dotenv import load_dotenv
from time import time

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
from utils.prompts import translator_prompt_qwen, formatting_headers_prompt, translator_prompt_template, headers_cleaning_prompt, signature_cleaning_prompt,extraction_prompt
from agents.preprocessing_agent import clean_email_llm, extract_email_llm
from agents.translator_agent import translate_email_llm


if __name__ == "__main__":
    tic1 = time()
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    #model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    #model_name = "Qwen/Qwen2.5-14B-Instruct"
    model_name2 = "LuvU4ever/qwen2.5-3b-qlora-merged-v4"
    #model_name2 = "facebook/seamless-m4t-v2-large"
    #model_name2 = "Helsinki-NLP/opus-mt-mul-en"
    #model_name2 = "LLaMAX/LLaMAX2-7B-Alpaca"
    #model_name2 = "winninghealth/WiNGPT-Babel-2-AWQ"

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device0 = "cuda:2" if num_gpus > 1 else "cuda:0"
        device1 = "cuda:3" if num_gpus > 2 else "cuda:0"
    else:
        device0 = device1 = "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map="auto",
        max_memory={
            0: "16GB",   # allow GPU 0
            1: "16GB"    # allow GPU 1
        }
    )#.to(device0)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # model2 = AutoModelForCausalLM.from_pretrained(
    #     model_name2,
    #     torch_dtype=torch.float16,
    #     attn_implementation="sdpa",
    #     #device_map="auto",
    # ).to(device1)

    # tokenizer2 = AutoTokenizer.from_pretrained(model_name2)

    email_text = """Στις 27/12/2023 1:31 μ.μ., ο/η Mairy Meni έγραψε:
Καλησπερα σας , Χρονια πολλα 
θα επανελθουμε με τα στοιχεια της κρατησης  για το εν θεματι φορτιο 
IMPORTANT 
Kind regards
Mairy Meni (Ms.)
Export Operations Department
Arian Maritime S.A.
133A Filonos street | Piraeus-Greece 18536
"""

    formatted_email = clean_email_llm(email_text, prompt=formatting_headers_prompt, model=model, tokenizer=tokenizer, trace_name="format_email_headers", device=device0)
    print("\n\nFormatted email:\n", formatted_email)
    LOGGER.info(f"Time taken to process: {time() - tic1} seconds")

    translated_email = clean_email_llm(formatted_email, prompt=translator_prompt_template , model=model, tokenizer=tokenizer, trace_name="translate_format_email", device=device0)
    print("\n\nTranslated email:\n", translated_email)
    LOGGER.info(f"Time taken to process: {time() - tic1} seconds") 

    cleaned_from_signatures = clean_email_llm(translated_email, prompt=signature_cleaning_prompt, model=model, tokenizer=tokenizer, trace_name="clean_email_signatures", device=device0)
    print("\n\nCleaned signatures:\n", cleaned_from_signatures)
    LOGGER.info(f"Time taken to process: {time() - tic1} seconds")

    cleaned_from_headers = clean_email_llm(cleaned_from_signatures, prompt=headers_cleaning_prompt, model=model, tokenizer=tokenizer, trace_name="clean_email_headers", device=device0)
    print("\n\nCleaned headers:\n", cleaned_from_headers)
    LOGGER.info(f"Time taken to process: {time() - tic1} seconds")

    # final_email = cleaned_from_headers.replace("Body:", "\n", 1)
    # print("\n\nFinal cleaned email:\n", final_email)

    msg = message_from_string(cleaned_from_headers)
    email_dict = {
    "from": msg["From"],
    "sent" :  msg["Sent"],
    "to": msg["To"],
    "cc" : msg["Cc"],
    "subject": msg["Subject"],
    "body": msg.get_payload()
    }
    #extracted_info = extract_email_llm(cleaned_from_headers, prompt=extraction_prompt, model=model, tokenizer=tokenizer, trace_name="email1", device=device0)
    print("\n\nExtracted Information:\n", json.dumps(email_dict, indent=2, ensure_ascii=False, default=str))
    LOGGER.info(f"Time taken to process: {time() - tic1} seconds")