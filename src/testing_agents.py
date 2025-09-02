import os, json
import torch
import re
from email import message_from_string
from transformers import pipeline
from dotenv import load_dotenv
from time import time

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
if langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
    os.environ["LANGSMITH_PROJECT"] = "email_preprocessing"
else:
    LOGGER.warning("Langsmith API key not found. Tracing will be disabled.")

from langsmith import traceable, trace, Client
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, AutoModelForSeq2SeqLM
from utils.logging_config import LOGGER
from utils.prompts import translator_prompt_qwen, formatting_headers_prompt, headers_cleaning_prompt, signature_cleaning_prompt,extraction_prompt
from agents.preprocessing_agent import clean_email_llm, extract_email_llm
from agents.translator_agent import translate_email_llm


if __name__ == "__main__":
    tic1 = time()
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model_name2 = "LuvU4ever/qwen2.5-3b-qlora-merged-v4"
    #model_name2 = "facebook/seamless-m4t-v2-large"
    #model_name2 = "Helsinki-NLP/opus-mt-mul-en"
    #model_name2 = "LLaMAX/LLaMAX2-7B-Alpaca"
    #model_name2 = "winninghealth/WiNGPT-Babel-2-AWQ"

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device0 = "cuda:1" if num_gpus > 1 else "cuda:0"
        device1 = "cuda:2" if num_gpus > 2 else "cuda:0"
    else:
        device0 = device1 = "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        #device_map="auto"
    ).to(device0)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model2 = AutoModelForCausalLM.from_pretrained(
        model_name2,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        #device_map="auto",
    ).to(device1)

    tokenizer2 = AutoTokenizer.from_pretrained(model_name2)

    email_text = """From: "LYNN@GLOBALSAILING.COM.CN" <LYNN@GLOBALSAILING.COM.CN>
Sent: 2023-12-29 14:09:28
To: sofia <sofia@arianmaritime.gr>,Vicky Parissi <operations01@arianmaritime.gr>
Subject: 234107 1x20dv QINGDAO / THESSALONIKI S/ROSING C/ZAK
Hi Sofia,
This shipment space is confirmed by HMM,i will send booking confirmation to you as soon as i got it,thanks~
Thanks and Best regards
Lynn Liu   刘小姐  
Import and Export Specialist
Hangzhou Global Sailing Shipping Co., Ltd. (杭州世航船务有限公司)
*mobile/WhatsApp  +86 151 5886 2437
*direct line. +86 571-8770 1148
*email. lynn@globalsailing.com.cn <mailto:lynn@globalsailing.com.cn> 
skype. lynn2022@163.com <mailto:lynn2022@163.com> 
WhatsApp. +86 151 5886 2437
WCA ID.63100 
OLO ID:04372 
NVOCC. MOC-NV01262 
Website. www.globalsailing.com.cn 
Room 804,Building A,Huahong Plaza,No 238.Tian Mu Shan Road,
West Lake District,Hangzhou 310007 China
***TO AVOID HACKER,PLS RECONFIRM BANK ACCOUNT WITH ME VIA WECHAT OR SKYPE BEFORE ARRANGE PAYMENT***
**HOLIDAY NOTICE: Our office will be closed on 30th Dec- 1st Jan for New Year's Day and resumes to work on 2nd Jan**
**HOLIDAY NOTICE: Our office will be closed on 10th Feb- 17th Feb for Chinese New Year and resumes to work on 18th Feb**
寄文件到我司请一定不要用韵达，谢谢！
提单一律寄顺丰到付，开票只能开0税率电子发票，均不能抵扣， 如因贵司给错开票资料导致重开发票需要100元重开发票费，付款请公对公，不接受私对公转账，谢谢！
如各位工厂的货物是重货或裸装类货物，装箱时请自行拍照留底，后续产生坏污箱与我司无关，谢谢！
请各位工厂务必不要瞒报、谎报、漏报和误报货物品名或夹带未经申报的货物，尤其是任何种类的防疫物资和电池,化工品,危险品,仿牌货，如因此产生任何费用或扣货或者刑事责任，请工厂自行承担，与我司无关,谢谢！
各位工厂务必注意:如贵司未收齐货款请务必在出货前书面通知，如贵司货物到港前一周前还未收齐货款请务必书面通知我们控货，否则我司不付任何法律责任！！！
Please reconfirm with cnee before transfer any kinds of shipment.
If any rejection of goods due to coronavirus affect,our company will not pay detention&demurrage charges and legal responsibility for it.
If above message is useless or distrubing you, please click here at "lynn@globalsailing.com.cn <mailto:lynn@globalsailing.com.cn> " 
for delete, we will exclude your personal information from our list within 10working days. Highly appreciated for your coopertion. thanks
P Please consider your environmental responsibility before printing this e-mail.
"""

    translated_email = translate_email_llm(email_text, prompt=translator_prompt_qwen, model=model2, tokenizer=tokenizer2, trace_name="translate_format_email", device=device1)
    print("\n\nTranslated output:\n", translated_email)
    LOGGER.info(f"Time taken to process: {time() - tic1} seconds") 

    formatted_email = clean_email_llm(translated_email, prompt=formatting_headers_prompt, model=model, tokenizer=tokenizer, trace_name="format_email_headers", device=device0)
    print("\n\nFormatted Email:\n", formatted_email)
    LOGGER.info(f"Time taken to process: {time() - tic1} seconds")

    cleaned_from_signatures = clean_email_llm(formatted_email, prompt=signature_cleaning_prompt, model=model, tokenizer=tokenizer, trace_name="clean_email_signatures", device=device0)
    print("\n\nCleaned signatures:\n", cleaned_from_signatures)
    LOGGER.info(f"Time taken to process: {time() - tic1} seconds")

    cleaned_from_headers = clean_email_llm(cleaned_from_signatures, prompt=headers_cleaning_prompt, model=model, tokenizer=tokenizer, trace_name="clean_email_headers", device=device0)
    print("\n\nCleaned headers:\n", cleaned_from_headers)
    LOGGER.info(f"Time taken to process: {time() - tic1} seconds")

    msg = message_from_string(cleaned_from_headers)
    email_dict = {
    "from": msg["From"],
    "to": msg["To"],
    "cc" : msg["Cc"],
    "subject": msg["Subject"],
    "body": msg.get_payload()
    }
    # extracted_info = extract_email_llm(cleaned_from_headers, prompt=extraction_prompt, model=model, tokenizer=tokenizer, trace_name="email1", device=device0)
    print("\n\nExtracted Information:\n", json.dumps(email_dict, indent=2, ensure_ascii=False, default=str))
    # LOGGER.info(f"Time taken to process: {time() - tic1} seconds")