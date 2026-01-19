import os, json, sys
from concurrent.futures import ThreadPoolExecutor
from langsmith import traceable
from dotenv import load_dotenv
from time import time
from email import message_from_string
from typing import List
from openai import OpenAI

from utils.logging import LOGGER
from utils.email_utils import extract_msg_file, clean_data, split_email_thread
from ..prompts.preprocessing_prompts import cleaning_prompt, formatter_and_translator_prompt

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
            base_url=os.getenv("LLM_BINDING_HOST"),
            api_key=os.getenv("LLM_BINDING_API_KEY"),
            max_retries=3
        )
        self.model = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8")

    @traceable
    def process_single_prompt(self, prompt:str)->str:
        """Processes a single prompt using the standard completions API."""
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            temperature=0,
            max_tokens=2048,
            stop="End of email"
        )

        return response.choices[0].text

    def __call__(self, prompt_list: List[str])->List[dict]:
        preprocessed_emails = []
        
        with ThreadPoolExecutor(max_workers=6) as executor:
            generated_texts_iterator = list(executor.map(self.process_single_prompt, prompt_list))

            for generated_text in generated_texts_iterator:
                try:
                    # generated_text is the string result from process_single_prompt
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
                    
                except Exception as e:
                    LOGGER.error(f"Structured extraction failed: {e}")
                   
        return preprocessed_emails
        
def main():
    tic1 = time()
    if len(sys.argv) != 2:
        LOGGER.error("Usage: python vllm_predictor.py <dir_path>")
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
    formatting_prompts = [formatter_and_translator_prompt.format(email=e) for e in all_emails_to_process]
    results = predictor(formatting_prompts)

    str_results = [str(r) for r in results]

    cleaning_prompts = [cleaning_prompt.format(email=e) for e in str_results]
    results = predictor(cleaning_prompts)


    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False, default=str)

    LOGGER.info(f"Time taken to process: {time() - tic1} seconds")


if __name__ == "__main__":
    main()