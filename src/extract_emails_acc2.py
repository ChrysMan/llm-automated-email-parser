import json, sys, os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:128"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import math
import torch
import time
import torch.distributed as dist
from accelerate import PartialState, infer_auto_device_map
from accelerate.utils import gather_object
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    pipeline, 
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList
)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFacePipeline
from datetime import timedelta
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from utils.logging_config import LOGGER
from utils.graph_utils import extract_msg_file, clean_data, split_email_thread, write_file, append_file

# Configure torch memory settings
torch.backends.cuda.cufft_plan_cache.clear()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

# Environment configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
langsmith_api_key = os.environ.get("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

if not langsmith_api_key:
    LOGGER.warning("Langsmith API key not found. Tracing will be disabled.")

# Constants
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_EMAILS_PER_PROMPT = 4  # Max emails per context batch
MAX_SEQUENCE_LENGTH = 8192  # Llama 3 context limit

class EmailInfo(BaseModel):
    """Data model for email extracted information."""
    sender: str = Field(..., description="The sender of the email. Store their name and email, if available.")
    sent: str = Field(..., description="The date the email was sent.")
    to: Optional[List[str]] = Field(default_factory=list, description="A list o recipients' names and emails.")
    cc: Optional[List[str]] = Field(default_factory=list, description="A list of additional recipients' names and emails.")
    subject: Optional[str] = Field(None, description="The subject of the email.")
    body: str = Field(..., description="The email body, excluding unnecessary whitespace.")

parser = JsonOutputParser(pydantic_object=EmailInfo, json_compatible=True)

prompt_template = PromptTemplate.from_template(
"""<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are an expert assistant who processes email threads with precision and no loss of content.
The input consists of several email strings. Some emails may be duplicates. Your task is:
1. Split the email thread into individual emails using "***" as the only delimiter. Each "***" marks the end of one complete email.
2. Identify and remove any duplicate emails in the list based on content mainting the chronological order.
3. Return the unique emails **ONLY** as a JSON list, where each email is a **separate object** with the following fields:

- sender: The sender of the email. Store their name and email, if available, as a string in the format "Name <email@example.com>". If only a name is present, store it as "Name". 
- sent: Extract and include **only the date and time**. 
- to: A list of recipients' names and emails, if available. Store each entry as a string in the format "Name <email@example.com>" or "Name". The "To" field may contain multiple recipients separated by commas or semicolons but is usually one.
- cc: A list of additional recipients' names and emails. Store each entry as a string in the format "Name <email@example.com>" or "Name". The "Cc" field may contain multiple recipients separated by commas or semicolons.
- subject: The subject of the email, stored as a string. Extract this after the "Subject:" field.
- body:  Include the full content of the message starting from the line **after** "Subject:" or "wrote:", and continue **until the next delimiter**. Do not summarize or skip any content.

4. Maintain the chronological order of the emails in the output.
5. Do not hallucinate or add any information that is not present in the email thread.
6. The length of the JSON list needs to be the same as the number of the emails strictly.
7. Output ONLY the raw JSON array. Do NOT include extra quotes, explanations, or any text.

Process the following email thread:
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
{emails}
<|eot_id|>

<|start_header_id|>assistant<|end_header_id>
"""
)

#local_rank = int(os.environ.get("LOCAL_RANK", 0))
#torch.cuda.set_device(local_rank)

# Initialize distributed state
state = PartialState()

class ContextAwareBatchProcessor:
    def __init__(self, max_batch_size: int = 1):
        """
        max_batch_size: How many email batches to process simultaneously on one GPU
        """
        self.max_batch_size = max_batch_size
        self._init_model()

    def _init_model(self):
        """Initialize model with optimized settings for long sequences"""
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
            #bnb_4bit_use_double_quant=True,
            #llm_int8_skip_modules=["lm_head"],
            #bnb_4bit_quant_storage=torch.uint8,
            #load_in_4bit_skip_modules=["lm_head"] 

        )
        
        '''
        device_map = infer_auto_device_map(
            model=meta_model,
            no_split_module_classes=["LlamaDecoderLayer"],
            max_memory={i: "14GiB" for i in range(8)},
            dtype=torch.bfloat16,
            offload_buffers=False,  # Keep buffers on GPU
            fallback_allocation=True  # Allow overflow to CPU if needed
        )
        '''

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.model_max_length = MAX_SEQUENCE_LENGTH
        
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map={"": local_rank},
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            #attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            offload_state_dict=True
            #offload_folder="offload",  # Directory for CPU offloading
            #max_memory={i: "12GiB" for i in range(8)}  # Reduce to 12GB/GPU
        )

        
        '''
        model.config.use_cache = False
        model.eval() 
        model.gradient_checkpointing_enable()
        '''

        #Stopping criteria for Llama 3 tokens
        class StopOnTokens(StoppingCriteria):
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                stop_ids = [128001, 128009]  # <|end_of_text|> and <|eot_id|>
                return any(stop_id in input_ids[0] for stop_id in stop_ids)

        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,  # Increased for longer responses
            batch_size=self.max_batch_size,
            return_full_text=False,
            do_sample=False,
            temperature=None,
            top_k=None,
            top_p=None,
            stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
            eos_token_id=[self.tokenizer.eos_token_id, 128009]
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        self.chain = SPLIT_EMAILS_CHAIN = (prompt_template | llm | parser)

    def invoke(self, *args, **kwargs):
        return self.chain.invoke(*args, **kwargs)

def distribute_email_processing(email_batches: List[str]) -> List[dict]:
    '''Distribute email batches across GPUs with dynamic scaling'''
    # Calculate resource allocation
    total_batches = len(email_batches)
    gpus_available = state.num_processes
    
    # Assign batches to current GPU
    batches_per_gpu = math.ceil(total_batches / gpus_available)
    start_idx = state.local_process_index * batches_per_gpu
    end_idx = min(start_idx + batches_per_gpu, total_batches)
    local_batches = email_batches[start_idx:end_idx]
    
    if not local_batches:
        return []
    
    # Process batches
    results = []
    for batch in local_batches:
        try:
            processor = ContextAwareBatchProcessor(max_batch_size=MAX_EMAILS_PER_PROMPT)
            result = processor.invoke(
                {"emails": batch},
                config={
                    "metadata": {"batch_size": len(batch.split("***"))},
                    "run_name": f"batch_{start_idx}"
                }
            )
            results.extend(result)
        except Exception as e:
            LOGGER.error(f"Batch processing failed: {str(e)}")
            # Add placeholder for failed batch
            results.extend([{}] * len(batch.split("***")))
    
    return results

def split_and_extract_emails_acc(file_path: str) -> List[Dict]:
    """Distributed email processing with context-preserving batches"""
    os.environ["LANGCHAIN_PROJECT"] = "extract_emails_acc"
    
    raw_msg_content = extract_msg_file(file_path)
    cleaned_msg_content = clean_data(raw_msg_content)
    LOGGER.info("Splitting emails...")
    
    try:
        splitted_emails = split_email_thread(cleaned_msg_content)[::-1]
        
        # Create context-preserving batches
        email_batches = []
        for i in range(0, len(splitted_emails), MAX_EMAILS_PER_PROMPT):
            batch = splitted_emails[i:i+MAX_EMAILS_PER_PROMPT]
            batch_str = "\n***\n".join(batch)
            email_batches.append(batch_str)
        
        LOGGER.info(f"Created {len(email_batches)} batches for processing")
        
        # Distribute processing
        local_results = distribute_email_processing(email_batches)
        
        state.wait_for_everyone()
        all_results = gather_object(local_results)
        
        if state.is_main_process:
            # Flatten and remove empty results
            flat_results = [item for sublist in all_results for item in sublist if item]
            LOGGER.info(f"Total output emails: {len(flat_results)}")
            
            return flat_results
    except Exception as e:
        LOGGER.error(f"Failed to split emails: {e}")
        return []

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if len(sys.argv) != 2:
        LOGGER.error("Usage: python emailParsing.py <dir_path>")
        sys.exit(1)
    
    dir_path = sys.argv[1]
    folder_name = os.path.basename(os.path.normpath(dir_path))
    output_path = os.path.join(dir_path, f"{folder_name}.json")

    email_data = []
    for filename in os.listdir(dir_path):
        if filename.endswith(".msg"):
            file_path = os.path.join(dir_path, filename)
            try:
                tic = time.time()
                data = split_and_extract_emails_acc(file_path)
                LOGGER.info(f"Time taken to process {filename}: {time.time() - tic:.2f} seconds")
                email_data.extend(data) 
            except Exception as e:
                LOGGER.error(f"Processing {filename} failed: {e}")
    
    if state.is_main_process or not state.use_distributed:
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(email_data, file, indent=4, ensure_ascii=False, default=str)
        LOGGER.info(f"Results saved to {output_path}")