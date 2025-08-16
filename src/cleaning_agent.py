import os
from time import time
from langsmith import traceable
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.logging_config import LOGGER
from utils.graph_utils import clean_data
from utils.prompts import create_FewShotPrompt, headers_cleaning_examples, headers_cleaning_prefix, signature_cleaning_prompt

@traceable(name="SE-230054-7")
def clean_email(email_text:str, prompt) -> str:
    """Cleans the email text by removing unnecessary information and formatting."""
    try:
        # Prepare the prompt
        prompt_text = prompt.format(email=email_text)
        print("Prompt text:\n", prompt_text)

        # Tokenize
        input = tokenizer(prompt_text, return_tensors="pt")
        input = input.to('cuda')  # Move to the correct device

        token_ids = tokenizer.encode(email_text)
        token_count = len(token_ids)
        max_new_tokens = next_power_of_two(token_count)
        print(f"Token count: {token_count}, Max new tokens: {max_new_tokens}\n\n")
        print("<|eot_id|>: ", tokenizer.encode("<|eot_id|>", add_special_tokens=False))
        print("\n\n")
        
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
        #print("Raw response:\n", cleaned_email_text)
        real_response = cleaned_email_text.split("<|start_header_id|>assistant<|end_header_id>")[-1].split("---End of email---")[0].strip()
        cleaned_response = clean_data(real_response)
        return cleaned_response
    except Exception as e:
        LOGGER.error(f"Error cleaning email: {e}")
        return email_text  # Return original if error occurs
    
def next_power_of_two(x: int) -> int:
    if x <= 0:
        return 1
    return 1 << (x - 1).bit_length()

if __name__ == "__main__":
    langsmith_api_key = os.environ.get("LANGSMITH_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
    os.environ["LANGSMITH_PROJECT"] = "email_cleaning"
    if not langsmith_api_key:
        LOGGER.warning("Langsmith API key not found. Tracing will be disabled.")

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="float16",
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    email_text = """Your email text goes here."""

    headers_cleaning_prompt = create_FewShotPrompt(headers_cleaning_examples, headers_cleaning_prefix)

    cleaned__from_headers = clean_email(email_text, prompt=headers_cleaning_prompt)
    print("Cleaned headers: \n", cleaned__from_headers)
    cleaned_from_signatures = clean_email(cleaned__from_headers, prompt=signature_cleaning_prompt)
    print("Cleaned signatures: \n", cleaned_from_signatures)