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
        self.llm = LLM(
            model="Qwen/Qwen2.5-14B-Instruct",
            tensor_parallel_size=2,
            trust_remote_code=True,
            enforce_eager=True,
            dtype='float16',
            gpu_memory_utilization=0.7,
            cpu_offload_gb=3,
            max_model_len=8192
        )

        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=4096,
            stop="End of email",
            detokenize=True
        )
        
    @traceable
    def __call__(self, prompt_list: List[str])->List[dict]:
        outputs = self.llm.generate(prompt_list, self.sampling_params)

        preprocessed_emails = []

        for output in outputs:
            msg = message_from_string(output)
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

    preprocessed_emails = []
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

            prompts = [overall_cleaning_prompt.format(email=e) for e in splitted_emails]

            preprocessed_emails.append(predictor(prompts))
            LOGGER.info(f"Time taken to process {filename}: {time() - tic2} seconds")

    predictor.llm.shutdown()

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(preprocessed_emails, file, indent=4, ensure_ascii=False, default=str)

    LOGGER.info(f"Time taken to process: {time() - tic1} seconds")


if __name__ == "__main__":
    main()