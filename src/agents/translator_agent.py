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
from utils.logging_config import LOGGER
from utils.graph_utils import clean_data

client = Client()

@traceable
def translate_email_llm(email_text: str, prompt:str, model:AutoModelForCausalLM, tokenizer: AutoTokenizer, trace_name:str, device:torch.device) -> str:
    try:
        with trace(
                name=f"{trace_name}",
                metadata={
                    "model_name": model.name_or_path
                }
            ):
            #for facebook/m2m100_418M
            # tokenizer.src_lang = "el"
            # encoded_zh = tokenizer(email_text, return_tensors="pt")
            # encoded_zh = encoded_zh.to(device)
            # generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id("en"), max_new_tokens=encoded_zh["input_ids"].shape[-1])
            # output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            # for LuvU4ever/qwen2.5-3b-qlora-merged-v4 and winninghealth/WiNGPT-Babel-2-GGUF
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": email_text}
            ]
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)
            
            #print(f"Token ID: {inputs["input_ids"].max()}, model vocab size:  {model.config.vocab_size}")
            #assert inputs["input_ids"].max() < model.config.vocab_size, f"Token ID exceeds model vocab size: {inputs["input_ids"].max()}, {model.config.vocab_size}"
            outputs = model.generate(**inputs, do_sample=False, max_new_tokens=inputs["input_ids"].shape[-1]) #inputs["input_ids"].shape[-1]
            output = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            #output = "From: "+ output.split("From:", 1)[1].strip() if "From:" in output else output.strip()
            #cleaned_response = clean_data(output)
            return output
    except Exception as e:
        LOGGER.error(f"Error translating email: {e}")
        return ""  # Return original if error occurs
