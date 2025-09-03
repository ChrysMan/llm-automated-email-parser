import os
import torch
from dotenv import load_dotenv

load_dotenv()

langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
if langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
    os.environ["LANGSMITH_PROJECT"] = "email_preprocessing"
else:
    LOGGER.warning("Langsmith API key not found. Tracing will be disabled.")

from langsmith import traceable, trace, Client
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.output_parsers import JsonOutputParser
from utils.logging_config import LOGGER
from utils.prompts import EmailInfo, extraction_prompt
from utils.graph_utils import clean_data

client = Client()

parser = JsonOutputParser(pydantic_object=EmailInfo, json_compatible=True)

@traceable
def clean_email_llm(email_text:str, prompt, model:AutoModelForCausalLM, tokenizer: AutoTokenizer, trace_name:str, device:torch.device) -> str:
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

            # Generate the cleaned email text
            cleaned_email = model.generate(
                input_ids=input['input_ids'],
                attention_mask=input['attention_mask'],
                max_new_tokens=input['input_ids'].shape[-1],
                do_sample=False,
                pad_token_id = tokenizer.eos_token_id
            )

            # Decode the generated text
            cleaned_email_text = tokenizer.decode(cleaned_email[0], skip_special_tokens=False)
            #print(cleaned_email_text)
            # Extract the relevant part of the response
            real_response = cleaned_email_text.split("<|start_header_id|>assistant<|end_header_id>")[-1].split("---End of email---")[0].strip()
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

