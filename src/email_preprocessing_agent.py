import json, os, sys
from time import time

import torch
from utils.logging_config import LOGGER
from email import message_from_string
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.graph_utils import extract_msg_file, clean_data, split_email_thread
from agents.preprocessing_agent import clean_email_llm
from utils.prompts import cleaning_prompt, formatting_headers_prompt, translator_prompt_template, overall_cleaning_prompt

if __name__ == "__main__":

    tic1 = time()

    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    if len(sys.argv) != 2:
        LOGGER.error("Usage: python emailParsing.py <dir_path>")
        sys.exit(1)

    dir_path = sys.argv[1]
    if not os.path.isdir(dir_path):
        LOGGER.error(f"{dir_path} is not a valid directory.")
        sys.exit(1)

    folder_name = os.path.basename(os.path.normpath(dir_path))

    output_path = os.path.join(dir_path, f"{folder_name}.json")

    #model_tag = "meta-llama/Llama-3.1-8B-Instruct"
    model_name = "Qwen/Qwen2.5-14B-Instruct"
    #model_name2 = "LuvU4ever/qwen2.5-3b-qlora-merged-v4"

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device0 = "cuda:0" if num_gpus > 1 else "cuda:0"
        device1 = "cuda:2" if num_gpus > 2 else "cuda:0"
    else:
        device0 = device1 = "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        attn_implementation="sdpa",
        device_map="auto",
        max_memory={
            1: "16GB",  # allow GPU 0
            2: "16GB",   # allow GPU 1
            3: "16GB"    # allow GPU 2
        }
    )#.to(device0)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    email_data = []

    for filename in os.listdir(dir_path):
        if filename.endswith(".msg"):
            file_path = os.path.join(dir_path, filename)

            tic2 = time()

            try:
                #  with open("/home/chryssida/src/Texts/AE-230009-split.txt", "a") as f:
                raw_msg_content = extract_msg_file(file_path)
                cleaned_msg_content = clean_data(raw_msg_content)
                splitted_emails = split_email_thread(cleaned_msg_content)

                    # joined = "\n-***-\n".join(splitted_emails)
                    # f.write(joined)
            except Exception as e:
                LOGGER.error(f"Failed to extract or clean email from {filename}: {e}")
                continue

            try:
                count = 0
                for email in splitted_emails:
                    count += 1
                    
                    cleaned_email = clean_email_llm(email, prompt=overall_cleaning_prompt, model=model, tokenizer=tokenizer, trace_name=f"clean_email_{filename}_{count}")
                    # formatted_email = clean_email_llm(email, prompt=formatting_headers_prompt, model=model, tokenizer=tokenizer, trace_name=f"format_email_headers_{filename}_{count}", device=device0)
                    # translated_email = clean_email_llm(formatted_email, prompt=translator_prompt_template, model=model, tokenizer=tokenizer, trace_name=f"translate_{filename}_{count}", device=device0)
                    # cleaned_from_signatures_headers = clean_email_llm(translated_email, prompt=cleaning_prompt, model=model, tokenizer=tokenizer, trace_name=f"clean_email_{filename}_{count}", device=device0)
                    
                    msg = message_from_string(cleaned_email)
                    email_dict = {
                    "from": msg["From"],
                    "sent": msg["Sent"],
                    "to": msg["To"],
                    "cc" : msg["Cc"],
                    "subject": msg["Subject"],
                    "body": msg.get_payload()
                    }
                    email_data.append(email_dict)
            except Exception as e:
                LOGGER.error(f"Failed to clean or extract email from {filename}: {e}")
                continue

            LOGGER.info(f"Time taken to process {filename}: {time() - tic2} seconds")
        
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(email_data, file, indent=4, ensure_ascii=False, default=str)

    LOGGER.info(f"Time taken to process: {time() - tic1} seconds")
    