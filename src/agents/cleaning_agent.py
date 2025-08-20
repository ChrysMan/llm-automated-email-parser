import os
import torch

langsmith_api_key = os.environ.get("LANGSMITH_API_KEY")
if langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
    os.environ["LANGSMITH_PROJECT"] = "email_preprocessing"
else:
    LOGGER.warning("Langsmith API key not found. Tracing will be disabled.")

from langsmith import traceable, trace, Client
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.logging_config import LOGGER
from utils.graph_utils import clean_data
from utils.prompts import headers_cleaning_prompt, signature_cleaning_prompt

client = Client()

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
            #print("Prompt text:\n", prompt_text)

            # Tokenize
            input = tokenizer(prompt_text, return_tensors="pt")
            input = input.to(device)  # Move to the correct device

            token_ids = tokenizer.encode(email_text)
            token_count = len(token_ids)

            # Calculate max_new_tokens based on the token count. Ensure it is a power of two.
            if token_count <= 0:
                max_new_tokens =  1
            max_new_tokens = 1 << (token_count - 1).bit_length()
            
            # Generate the cleaned email text
            cleaned_email = model.generate(
                input_ids=input['input_ids'],
                attention_mask=input['attention_mask'],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id = tokenizer.eos_token_id
            )
            
            # Decode the generated text
            cleaned_email_text = tokenizer.decode(cleaned_email[0], skip_special_tokens=False)

            # Extract the relevant part of the response
            real_response = cleaned_email_text.split("<|start_header_id|>assistant<|end_header_id>")[-1].split("---End of email---")[0].strip()
            cleaned_response = clean_data(real_response)
            return cleaned_response
    except Exception as e:
        LOGGER.error(f"Error cleaning email: {e}")
        return email_text  # Return original if error occurs

if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="float16",
        attn_implementation="sdpa"
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    email_text = """Your email goes here"""

    cleaned_from_signatures = clean_email_llm(email_text, prompt=signature_cleaning_prompt, model=model, tokenizer=tokenizer, trace_name="email_signatures", device=device)
    print("\n\nCleaned signatures: \n", cleaned_from_signatures)
    cleaned__from_headers = clean_email_llm(cleaned_from_signatures, prompt=headers_cleaning_prompt, model=model, tokenizer=tokenizer, trace_name="email_headers", device=device)
    print("\n\nCleaned headers: \n", cleaned__from_headers)
    