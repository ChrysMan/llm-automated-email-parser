import os, json, sys
from langsmith import traceable
from vllm import LLM, SamplingParams
from dotenv import load_dotenv
from time import time
from email import message_from_string
from typing import List

from utils.logging import LOGGER
from utils.email_utils import extract_msg_file, clean_data, split_email_thread
from lightrag_impl.prompts.preprocessing_prompts import cleaning_prompt, formatter_and_translator_prompt

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "0"

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
            model="Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8",
            tensor_parallel_size=2,
            trust_remote_code=True,
            enforce_eager=True,
            dtype='float16',
            gpu_memory_utilization=0.8,
            max_num_seqs=20,    #maximum number of active sequences (parallel inferences)
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
            generated_text = output.outputs[0].text
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
        LOGGER.error("Usage: python vllm.py <dir_path>")
        sys.exit(1)

    dir_path = sys.argv[1]
    if not os.path.isdir(dir_path):
        LOGGER.error(f"{dir_path} is not a valid directory.")
        sys.exit(1)

    folder_name = os.path.basename(os.path.normpath(dir_path))

    output_path = os.path.join(dir_path, f"{folder_name}_vllm.json")

    predictor = LLMPredictor()

    splitted_emails = []
    for filename in os.listdir(dir_path):
        if filename.endswith(".msg"):
            file_path = os.path.join(dir_path, filename)

            #tic2 = time()
            try:
            #  with open("/home/chryssida/src/Texts/AE-230009-split.txt", "a") as f:
                raw_msg_content = extract_msg_file(file_path)
                cleaned_msg_content = clean_data(raw_msg_content)
                splitted_emails.extend(split_email_thread(cleaned_msg_content))

                    # joined = "\n-***-\n".join(splitted_emails)
                    # f.write(joined)
            except Exception as e:
                LOGGER.error(f"Failed to extract or clean email from {filename}: {e}")
                continue

    formatting_prompts = [formatter_and_translator_prompt.format(email=e) for e in splitted_emails]
    results = predictor(formatting_prompts)

    str_results = [str(r) for r in results]

    cleaning_prompts = [cleaning_prompt.format(email=e) for e in str_results]
    results = predictor(cleaning_prompts)

    #LOGGER.info(f"Time taken to process {filename}: {time() - tic2} seconds")


    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False, default=str)

    LOGGER.info(f"Time taken to process: {time() - tic1} seconds")


if __name__ == "__main__":
    main()