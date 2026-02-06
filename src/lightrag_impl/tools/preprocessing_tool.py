import os, json, re
from time import time

from ..preprocessing.vllm_predictor import LLMPredictor
from ..preprocessing.deduplicate import deduplicate_emails
from ..prompts.preprocessing_prompts import cleaning_prompt,formatter_and_translator_prompt, formatter_prompt, translator_prompt, body_cleaning_prompt, headers_cleaning_prompt
from utils.email_utils import extract_msg_file, clean_data, split_email_thread
from utils.logging import LOGGER


def execute_full_preprocessing(dir_path: str)-> str:
    """Use this tool to preprocess emails in the given directory and return a list of cleaned and unique email texts."""
    tic = time()
    
    folder_name = os.path.basename(os.path.normpath(dir_path))
    
    output_path = os.path.join(dir_path, f"{folder_name}_unique_2pr-2.json")

    predictor = LLMPredictor()

    all_emails_to_process = []
    for filename in os.listdir(dir_path):
        if filename.endswith(".msg"):
            file_path = os.path.join(dir_path, filename)

            try:
                raw_msg_content = extract_msg_file(file_path)
                cleaned_msg_content = clean_data(raw_msg_content)
                all_emails_to_process.extend(split_email_thread(cleaned_msg_content))
                LOGGER.info(f"Extraction completed successfully")
            except Exception as e:
                LOGGER.error(f"Failed to extract or clean email from {filename}: {e}")
                return "Failed to extract or clean email from {filename}: {e}"

    # Prepare all prompts outside the file loop
    try:
        formatting_prompts = [formatter_and_translator_prompt.format(email=e) for e in all_emails_to_process]
        results = predictor(formatting_prompts)

        str_results = [str(r) for r in results]

        cleaning_prompts = [cleaning_prompt.format(email=e) for e in str_results]
        results = predictor(cleaning_prompts)

        # translator_prompts = [translator_prompt.format(email=e) for e in all_emails_to_process]
        # results = predictor(translator_prompts)

        # str_results = [str(r) for r in results]

        # formatting_prompts = [formatter_prompt.format(email=e) for e in str_results]
        # results = predictor(formatting_prompts)

        # str_results = [str(r) for r in results]

        # body_cleaning_prompts = [body_cleaning_prompt.format(email=e) for e in str_results]
        # results = predictor(body_cleaning_prompts)

        # str_results = [str(r) for r in results]

        # headers_cleaning_prompts = [headers_cleaning_prompt.format(email=e) for e in str_results]
        # results = predictor(headers_cleaning_prompts)

    except Exception as e:
        LOGGER.error(f"Error during LLM processing: {e}")
        return f"Error during LLM processing: {e}"

    # ---------------Deduplicate results---------------
    try:
        unique_emails = deduplicate_emails(results)
        emails_json = []
        for text in unique_emails:
                # Split at the first "body:" (case-insensitive, multi-line safe)
                parts = re.split(r'(?mi)^\s*body\s*:\s*', text, maxsplit=1)
                headers_part = parts[0]
                body_part = parts[1] if len(parts) == 2 else ""

                email_dict = {}
                for line in headers_part.splitlines():
                    if ":" in line:
                        key, val = line.split(":", 1)  # split only on first colon
                        email_dict[key.strip().lower()] = val.strip()

                email_dict["body"] = body_part.rstrip()
                emails_json.append(email_dict)

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(emails_json, file, indent=4, ensure_ascii=False, default=str)

        LOGGER.info(f"Deduplication completed successfully")
        LOGGER.info(f"The preprocessing has completed in {time() - tic} seconds and the output is saved at {output_path}.")
        return f"The preprocessing has completed in {time() - tic} seconds and the output is saved at {output_path}."
    except Exception as e:
        LOGGER.error(f"Error during deduplication or saving: {e}")
        return "Error during deduplication or saving: {e}"
   
