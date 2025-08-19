import os
from time import time
from langsmith import traceable
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.logging_config import LOGGER
from utils.graph_utils import clean_data
from utils.prompts import headers_cleaning_prompt, signature_cleaning_prompt

@traceable(name="email")
def clean_email_llm(email_text:str, prompt, tokenizer: AutoTokenizer) -> str:
    """Cleans the email text by removing unnecessary information and formatting."""
    try:
        # Prepare the prompt
        prompt_text = prompt.format(email=email_text)
        #print("Prompt text:\n", prompt_text)

        # Tokenize
        input = tokenizer(prompt_text, return_tensors="pt")
        input = input.to('cuda')  # Move to the correct device

        token_ids = tokenizer.encode(email_text)
        token_count = len(token_ids)

        # Calculate max_new_tokens based on the token count. Ensure it is a power of two.
        if token_count <= 0:
            max_new_tokens =  1
        max_new_tokens = 1 << (token_count - 1).bit_length()

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

        # Extract the relevant part of the response
        real_response = cleaned_email_text.split("<|start_header_id|>assistant<|end_header_id>")[-1].split("---End of email---")[0].strip()
        cleaned_response = clean_data(real_response)
        return cleaned_response
    except Exception as e:
        LOGGER.error(f"Error cleaning email: {e}")
        return email_text  # Return original if error occurs

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
        attn_implementation="sdpa",
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    email_text = """发件人: beth_szx@sztranslead.com 
发送时间: 2023-07-06 22:00
收件人: Athina Begka 
抄送: Anastasios Peppas ; Marina Koletzaki ; yolanda_szx ; Amy 
主题: 【SO details】ARM061: 232010 1X40 SHEKOU/PIRAEUS *S/LEO PAPER *C/GIOCHI PO#80612 & PO#81287/228814652 VIA MSK ETD:15-Jul
Dear Athina,
Good day to you.
dear,  SO details as below and pls kindly check it.
SO NO.:228814652
V/V: MSC OSCAR 327W
POL: SHEKOU
POD: PIRAEUS
Volume: 1*40'NOR
ETD Shekou: 15-Jul
SI cut offf: 12-Jul
CY cut off: 13-Jul
ETA Piraeus:7-Aug
O/F: usd1550/40'NOR subject to agency fee, 15 free days
Planned stuffing Date: are checking with shpr here
B/L instruction:
MBL: 
SHPR: SHENZHEN TIANQI TRADING COMPANY LTD 
CNEE & NOTIFY PARTY:
ARIAN MARITIME S.A.
133A, FILONOS STREET, 18536 PIRAEUS, GREECE
VAT : 997932152
HBL:
SHPR: as per shpr's instruction
CNEE: as per shpr's instruction
NOTIFY PARTY: as per shpr's instruction
Best Regards,
Beth
Sales 
Translead International Forwarding Co.,Ltd
深圳市中茂国际货运代理有限公司
SKYPE /"""

    cleaned__from_headers = clean_email_llm(email_text, prompt=headers_cleaning_prompt, tokenizer=tokenizer)
    #print("Cleaned headers: \n", cleaned__from_headers)
    cleaned_from_signatures = clean_email_llm(cleaned__from_headers, prompt=signature_cleaning_prompt, tokenizer=tokenizer)
    print("Cleaned signatures: \n", cleaned_from_signatures)