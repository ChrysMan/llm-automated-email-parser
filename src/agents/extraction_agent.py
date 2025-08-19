import os, json
from time import time
from langsmith import traceable
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.output_parsers import JsonOutputParser
from utils.logging_config import LOGGER
from utils.prompts import EmailInfo, extraction_prompt

parser = JsonOutputParser(pydantic_object=EmailInfo, json_compatible=True)

@traceable(name="email")
def extract_email_llm(email_text: str, prompt, tokenizer: AutoTokenizer) -> EmailInfo:
    """Extracts email information using a language model."""
    try:
        # Prepare the prompt
        prompt_text = prompt.format(email=email_text)
        
        # Tokenize
        input = tokenizer(prompt_text, return_tensors="pt")
        input = input.to('cuda')  # Move to the correct device

        token_ids = tokenizer.encode(email_text)
        token_count = len(token_ids)
        
        # Calculate max_new_tokens based on the token count. Ensure it is a power of two.
        if token_count <= 0:
            max_new_tokens =  1
        max_new_tokens = 1 << (token_count - 1).bit_length()

        # Generate the email information
        generated = model.generate(
            input_ids=input['input_ids'],
            attention_mask=input['attention_mask'],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
        input_length = input['input_ids'].shape[1]
        generated_tokens = generated[0][input_length:]

        # Decode the generated text
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
        # Parse the output as JSON
        email_info = parser.parse(generated_text)

        return email_info
    except Exception as e:
        LOGGER.error(f"Error extracting email: {e}")
        return EmailInfo()  # Return an empty EmailInfo object if error occurs

if __name__ == "__main__":
    langsmith_api_key = os.environ.get("LANGSMITH_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
    os.environ["LANGSMITH_PROJECT"] = "email_extraction"
    if not langsmith_api_key:
        LOGGER.warning("Langsmith API key not found. Tracing will be disabled.")

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="float16",
        attn_implementation="sdpa",
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    email_text = """From: beth_szx@sztranslead.com
Sent: 2023-07-06 22:00
To: Athina Begka
Cc: Anastasios Peppas; Marina Koletzaki; yolanda_szx; Amy
Subject: 【SO details】ARM061: 232010 1X40 SHEKOU/PIRAEUS *S/LEO PAPER *C/GIOCHI PO#80612 & PO#81287/228814652 VIA MSK ETD:15-Jul
Dear Athina,
Good day to you.
dear, SO details as below and pls kindly check it.
SO NO.:228814652
V/V: MSC OSCAR 327W
POL: SHEKOU
POD: PIRAEUS
Volume: 1*40'NOR
ETD Shekou: 15-Jul
SI cut off: 12-Jul
CY cut off: 13-Jul
ETA Piraeus:7-Aug
O/F: usd1550/40'NOR subject to agency fee, 15 free days
Planned stuffing Date: are checking with shpr here
Best Regards,
Beth"""

    extracted_info = extract_email_llm(email_text, prompt=extraction_prompt, tokenizer=tokenizer)
    print(f"Extracted Information:\n", json.dumps(extracted_info, indent=2, ensure_ascii=False, default=str))