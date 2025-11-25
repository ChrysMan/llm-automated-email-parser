import os, json, sys
from langsmith import traceable
from vllm import LLM, SamplingParams
from dotenv import load_dotenv
from time import time
from utils.logging_config import LOGGER
from email import message_from_string
from typing import Optional, List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.graph_utils import extract_msg_file, clean_data, split_email_thread
from agents.preprocessing_agent import clean_email_llm
from utils.prompts import cleaning_prompt, formatting_headers_prompt, translator_prompt_template, overall_cleaning_prompt
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
if langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
    os.environ["LANGSMITH_PROJECT"] = "email_preprocessing"
else:
    LOGGER.warning("Langsmith API key not found. Tracing will be disabled.")

class LLMPredictor:

    def __init__(self):
        self.client = OpenAI(
            base_url="http://localhost:8001/v1",
            api_key="EMPTY"
        )
        
    @traceable
    def process_single_prompt(self, prompt:str)->str:
        """Processes a single prompt using the standard completions API."""
        response = self.client.completions.create(
            model="Qwen/Qwen2.5-14B-Instruct",
            prompt=prompt,
            temperature=0,
            max_tokens=2048,
            stop="End of email"
        )

        return response.choices[0].text
    
    @traceable
    def __call__(self, prompt_list: List[str])->List[dict]:
        preprocessed_emails = []
        
        # FIX 1: Use ThreadPoolExecutor for concurrent/parallel API calls to vLLM.
        # This replaces the native vLLM batching. Max 8 concurrent workers is a safe start.
        with ThreadPoolExecutor(max_workers=8) as executor:
            # map applies process_single_prompt to every item in prompt_list
            outputs = list(executor.map(self.process_single_prompt, prompt_list))

        # FIX 2: Process all collected outputs (strings)
        for generated_text in outputs:
            # message_from_string requires a string, which is now correctly passed
            msg = message_from_string(generated_text) 
            
            email_dict = {
                    "from": msg["From"],
                    "sent": msg["Sent"],
                    "to": msg["To"],
                    "cc" : msg["Cc"],
                    "subject": msg["Subject"],
                    "body": msg.get_payload()
                    }
            preprocessed_emails.append(email_dict)

        return preprocessed_emails
        
def main():
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

    predictor = LLMPredictor()

    all_emails_to_process = []
    for filename in os.listdir(dir_path):
        if filename.endswith(".msg"):
            file_path = os.path.join(dir_path, filename)

            try:
                raw_msg_content = extract_msg_file(file_path)
                cleaned_msg_content = clean_data(raw_msg_content)
                all_emails_to_process.extend(split_email_thread(cleaned_msg_content))

            except Exception as e:
                LOGGER.error(f"Failed to extract or clean email from {filename}: {e}")
                continue

    # Prepare all prompts outside the file loop
    prompts = [overall_cleaning_prompt.format(email=e) for e in all_emails_to_process]

    results = predictor(prompts)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False, default=str)

    LOGGER.info(f"Time taken to process: {time() - tic1} seconds")


if __name__ == "__main__":
    main()