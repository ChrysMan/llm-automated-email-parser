import json, os, sys
from time import time
from utils.logging_config import LOGGER
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.graph_utils import extract_msg_file, clean_data, split_email_thread, chunk_emails,find_best_chunk_size
from agents.cleaning_agent import clean_email_llm
from agents.extraction_agent import extract_email_llm

if __name__ == "__main__":

    tic1 = time()

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
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="float16",
        attn_implementation="sdpa",
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    email_data = []

    for filename in os.listdir(dir_path):
        if filename.endswith(".msg"):
            file_path = os.path.join(dir_path, filename)

            tic2 = time()

            try:
                raw_msg_content = extract_msg_file(file_path)
                cleaned_msg_content = clean_data(raw_msg_content)
                splitted_emails = split_email_thread(cleaned_msg_content)
            except Exception as e:
                LOGGER.error(f"Failed to extract or clean email from {filename}: {e}")
                continue

            try:
                for email in splitted_emails:
                    cleaned_email = clean_email_llm(email, tokenizer=tokenizer, model=model)
                    email_info = extract_email_llm(cleaned_email, tokenizer=tokenizer, model=model)
                    email_data.append(email_info)
            except Exception as e:
                LOGGER.error(f"Failed to clean or extract email from {filename}: {e}")
                continue

            LOGGER.info(f"Time taken to process {filename}: {time() - tic2} seconds")
            
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(email_data, file, indent=4, ensure_ascii=False, default=str)

        LOGGER.info(f"Time taken to process: {time() - tic1} seconds")
      