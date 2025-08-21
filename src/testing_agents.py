import os
import torch

langsmith_api_key = os.environ.get("LANGSMITH_API_KEY")
if langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
    os.environ["LANGSMITH_PROJECT"] = "email_preprocessing"
else:
    LOGGER.warning("Langsmith API key not found. Tracing will be disabled.")

from langsmith import traceable, trace, Client
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, AutoModelForSeq2SeqLM
from utils.logging_config import LOGGER
from utils.prompts import translator_formatting_prompt, headers_cleaning_prompt, signature_cleaning_prompt
from agents.preprocessing_agent import clean_email_llm, extract_email_llm


if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model_name2 = "LLaMAX2-7B-Alpaca"

    device0 = "cuda:1" if torch.cuda.is_available() else "cpu"
    device1 = "cuda:2" if torch.cuda.is_available() else "cpu"
    
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype="float16",
    #     attn_implementation="sdpa",
    #     device_map="auto"
    # )#.to(device0)

    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    model2 = AutoModelForCausalLM.from_pretrained(
        model_name2,
        torch_dtype="float16",
        attn_implementation="sdpa",
        device_map="auto"
    )#.to(device1)

    tokenizer2 = AutoTokenizer.from_pretrained(model_name)

    email_text = """On Feb 15, 2023, at 13:20, Philemon Lerias wrote:
﻿ 
If the dims and cargo grw are accurate we can do 2050 eur / lumpsum . 
Shall we proceed ?
Please ensure you get updated information from our staff . 
Glad to be at your service. 
Best Regards,
Phil Lerias (Mr.)
Global Projects Manager – Sales
Arian Maritime S.A.
133A , Filonos Str | 18536 Piraeus – Greece 
Dir: +302381052351 | mob/viber/whatsapp : +306972281227
Skype : phil_lerias_uni 
Αγαπητοί Συνεργάτες , θα θέλαμε να σας ενημερώσουμε ότι, λόγω των έκτακτων μέτρων προφύλαξης από τον COVID-19, όσοι εισέρχονται στα γραφεία μας για παραλαβές -παραδόσεις  θα πρέπει υποχρεωτικά να φορούν μάσκα και γάντια μιας χρήσης.
Dear Cooperates , We hereby inform you that due to the COVID-19 health measures , it is mandatory to wear a face mask and medical gloves while in presence in the facilities of our Company.
"""
    formatted_email = clean_email_llm(email_text, prompt=translator_formatting_prompt, model=model2, tokenizer=tokenizer2, trace_name="translate_format_email", device=device1)
    print("\n\nTranslated and formatted email: \n", formatted_email)
    # cleaned_from_signatures = clean_email_llm(formatted_email, prompt=signature_cleaning_prompt, model=model, tokenizer=tokenizer, trace_name="clean_email_signatures", device=device0)
    # print("\n\nCleaned signatures: \n", cleaned_from_signatures)
    # cleaned_from_headers = clean_email_llm(cleaned_from_signatures, prompt=headers_cleaning_prompt, model=model, tokenizer=tokenizer, trace_name="clean_email_headers", device=device0)
    # print("\n\nCleaned headers: \n", cleaned_from_headers)

    # extracted_info = extract_email_llm(email_text, prompt=extraction_prompt, model=model, tokenizer=tokenizer, trace_name="email1", device=device)
    # print(f"Extracted Information:\n", json.dumps(extracted_info, indent=2, ensure_ascii=False, default=str))
    