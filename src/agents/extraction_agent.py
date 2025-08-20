import os, json
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
from langchain_core.output_parsers import JsonOutputParser
from utils.logging_config import LOGGER
from utils.prompts import EmailInfo, extraction_prompt

client = Client()

parser = JsonOutputParser(pydantic_object=EmailInfo, json_compatible=True)

@traceable
def extract_email_llm(email_text: str, prompt, model:AutoModelForCausalLM, tokenizer: AutoTokenizer, trace_name:str, device:torch.device) -> EmailInfo:
    """Extracts email information using a language model."""
    try:
        with trace(
                name=f"{trace_name}_generation",
                metadata={
                    "model_name": model.name_or_path
                }
            ):
            # Prepare the prompt
            prompt_text = prompt.format(email=email_text)
            
            # Tokenize
            input = tokenizer(prompt_text, return_tensors="pt")
            input = input.to(device)  # Move to the correct device

            token_ids = tokenizer.encode(email_text)
            token_count = len(token_ids)
            
            # Generate the email information
            generated = model.generate(
                input_ids=input['input_ids'],
                attention_mask=input['attention_mask'],
                max_new_tokens=token_count+128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
            input_length = input['input_ids'].shape[1]
            generated_tokens = generated[0][input_length:]

            #print(f"Token count: {token_count}, Max new tokens: {max_new_tokens}, Input length: {input_length}, Generated tokens: {len(generated_tokens)}")
            
            # Decode the generated text
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            #print("\nGenerated text:\n", generated_text)
            #real_response = generated_text.split("<|eot_id|>")[0].strip()

        with trace(
            name=f"{trace_name}_parsing",
            inputs={"prompt": generated_text},
            metadata={
                "model_name": model.name_or_path,
                "max_new_tokens": token_count + 128,
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
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="float16",
        attn_implementation="sdpa"
        #device_map="auto"
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    email_text = """Your email goes here"""

    extracted_info = extract_email_llm(email_text, prompt=extraction_prompt, model=model, tokenizer=tokenizer, trace_name="email1", device=device)
    print(f"Extracted Information:\n", json.dumps(extracted_info, indent=2, ensure_ascii=False, default=str))