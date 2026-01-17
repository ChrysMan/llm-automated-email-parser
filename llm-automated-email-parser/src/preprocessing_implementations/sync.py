import json, os, sys
import torch
from langsmith import traceable, trace
from time import time
from email import message_from_string
from langchain_core.output_parsers import JsonOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from lightrag_impl.prompts.preprocessing_prompts import EmailInfo, overall_cleaning_prompt
from utils.email_utils import extract_msg_file, clean_data, split_email_thread
from utils.logging import LOGGER

from dotenv import load_dotenv

load_dotenv()

langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
if langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
    os.environ["LANGSMITH_PROJECT"] = "email_preprocessing"
else:
    LOGGER.warning("Langsmith API key not found. Tracing will be disabled.")


parser = JsonOutputParser(pydantic_object=EmailInfo, json_compatible=True)

@traceable
def clean_email_llm(email_text:str, prompt, model:AutoModelForCausalLM, tokenizer: AutoTokenizer, trace_name:str) -> str:
    """Cleans the email text by removing unnecessary information and formatting."""
    try:
        with trace(
                name=f"{trace_name}",
                metadata={
                    "model_name": model.name_or_path
                }
            ):
            # Prepare the prompt
            prompt_text = prompt.format(email=email_text)

            # Tokenize
            input = tokenizer(prompt_text, return_tensors="pt").to('cuda')

            assert input['input_ids'].max() < model.config.vocab_size, f"Token ID exceeds model vocab size: {input['input_ids'].max()}, {model.config.vocab_size}"

            tokenizer.eos_token = "<|endoftext|>"

            gen_config = GenerationConfig(
                max_new_tokens=input['input_ids'].shape[-1],
                do_sample=False,
                eos_token_id = tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

            # Generate the cleaned email text
            cleaned_email = model.generate(
                input_ids=input['input_ids'],
                attention_mask=input['attention_mask'],
                generation_config=gen_config
            )

            # Decode the generated text
            cleaned_email_text = tokenizer.decode(cleaned_email[0], skip_special_tokens=False)
        
            # Extract the relevant part of the response
            real_response = cleaned_email_text.split("<|start_header_id|>assistant<|end_header_id>|")[-1].split("<|start_header_id|>assistant<|end_header_id>")[-1].split("End of email")[0].strip()
            cleaned_response = clean_data(real_response)
            return cleaned_response
    except Exception as e:
        LOGGER.error(f"Error cleaning email: {e}")
        return email_text  # Return original if error occurs

@traceable
def extract_email_llm(email_text: str, prompt, model:AutoModelForCausalLM, tokenizer: AutoTokenizer, trace_name:str, device:torch.device) -> EmailInfo:
    """Extracts email information using a language model."""
    try:
        with trace(
                name=f"{trace_name}",
                metadata={
                    "model_name": model.name_or_path
                }
            ):
            # Prepare the prompt
            prompt_text = prompt.format(email=email_text)
            
            # Tokenize
            input = tokenizer(prompt_text, return_tensors="pt").to(device)
            
            # Generate the email information
            generated = model.generate(
                input_ids=input['input_ids'],
                attention_mask=input['attention_mask'],
                max_new_tokens=input['input_ids'].shape[-1],
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

            # Decode the generated text
            generated_text = tokenizer.decode(generated[0][input['input_ids'].shape[-1]:], skip_special_tokens=True)
            #print("\nGenerated text:\n", generated_text)
            #real_response = generated_text.split("<|eot_id|>")[0].strip()

        with trace(
            name=f"{trace_name}",
            inputs={"prompt": generated_text},
            metadata={
                "model_name": model.name_or_path,
                "max_new_tokens": input['input_ids'].shape[-1],
                "generated_text_length": len(generated_text)
            }
        ):
            # Parse the output as JSON
            email_info = parser.parse(generated_text)

            return email_info
    except Exception as e:
        LOGGER.error(f"Error extracting email: {e}")
        return EmailInfo()  # Return an empty EmailInfo object if error occurs

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
    