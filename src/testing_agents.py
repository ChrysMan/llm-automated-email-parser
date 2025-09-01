import os, json
import torch
import re
from transformers import pipeline
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
from utils.prompts import translator_prompt_qwen, headers_cleaning_prompt, signature_cleaning_prompt,extraction_prompt
from agents.preprocessing_agent import clean_email_llm, extract_email_llm
from agents.translator_agent import translate_email_llm


if __name__ == "__main__":
    tic1 = time()
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model_name2 = "LuvU4ever/qwen2.5-3b-qlora-merged-v4"
    #model_name2 = "facebook/seamless-m4t-v2-large"
    #model_name2 = "Helsinki-NLP/opus-mt-mul-en"
    #model_name2 = "LLaMAX/LLaMAX2-7B-Alpaca"
    #model_name2 = "winninghealth/WiNGPT-Babel-2-AWQ"

    device0 = "cuda:0" if torch.cuda.is_available() else "cpu"
    device1 = "cuda:1" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        #device_map="auto"
    ).to(device0)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model2 = AutoModelForCausalLM.from_pretrained(
        model_name2,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        #device_map="auto",
    ).to(device1)

    tokenizer2 = AutoTokenizer.from_pretrained(model_name2)

    email_text = """From: Sofia Stafylaraki <sofia@arianmaritime.gr <mailto:sofia@arianmaritime.gr> > 
Sent: Τετάρτη, 13 Δεκεμβρίου 2023 4:28 μμ
To: 刘业 <lynn@globalsailing.com.cn <mailto:lynn@globalsailing.com.cn> >
Cc: Athina Begka <operations01@arianmaritime.gr <mailto:operations01@arianmaritime.gr> >
Subject: 1x20dv QINGDAO / THESSALONIKI S/ROSING C/ZAK
Dear Lynn
Proceed with this.
Athina 
Please add ref nr.
Thanks 
Best regards 
Sofia Stafilaraki (Mrs.)
Sales Executive
Please ensure you get updated information from our staff . 
Glad to be at your service. 
Θα θέλαμε να σας ενημερώσουμε ότι, λόγω των έκτακτων μέτρων προφύλαξης από τον COVID-19, όσοι εισέρχονται στα γραφεία μας για παραλαβές -παραδόσεις  θα πρέπει υποχρεωτικά να φορούν μάσκα και γάντια μιας χρήσης.Ενημερώστε παρακαλώ σχετικά εκτελωνιστές, υπαλληλους, εξωτερικους συνεργατες και λοιπούς να είναι προετοιμασμένοι ωστε να αποφύγουμε καθυστερήσεις και περαιτέρω αναστάτωση. 
Dear Cooperates , We hereby inform you that due to the COVID-19 health measures , it is mandatory to wear a face mask and medical gloves while in presence in the facilities of our Company. 
Arian Maritime S.A.
13, Mitropoleos street | 54624 Thessaloniki, Greece
Please be informed that as from Jan.. 27th ,2020 our PIRAEUS HQ offices will be located at : 
133A, FILONOS STR,  18536 PIRAEUS GREECE.
All other company details remain same.
Please update your files accordingly, for invoicing or sending us post mail."""

    translated_email = translate_email_llm(email_text, prompt=translator_prompt_qwen, model=model2, tokenizer=tokenizer2, trace_name="translate_format_email", device=device1)
    print("\n\nTranslated output:\n", translated_email)
    LOGGER.info(f"Time taken to process: {time() - tic1} seconds")

    cleaned_from_signatures = clean_email_llm(translated_email, prompt=signature_cleaning_prompt, model=model, tokenizer=tokenizer, trace_name="clean_email_signatures", device=device0)
    print("\n\nCleaned signatures: \n", cleaned_from_signatures)
    LOGGER.info(f"Time taken to process: {time() - tic1} seconds")

    cleaned_from_headers = clean_email_llm(cleaned_from_signatures, prompt=headers_cleaning_prompt, model=model, tokenizer=tokenizer, trace_name="clean_email_headers", device=device0)
    print("\n\nCleaned headers: \n", cleaned_from_headers)
    LOGGER.info(f"Time taken to process: {time() - tic1} seconds")

    extracted_info = extract_email_llm(cleaned_from_headers, prompt=extraction_prompt, model=model, tokenizer=tokenizer, trace_name="email1", device=device0)
    print(f"Extracted Information:\n", json.dumps(extracted_info, indent=2, ensure_ascii=False, default=str))
    LOGGER.info(f"Time taken to process: {time() - tic1} seconds")