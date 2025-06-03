import json, sys, os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch 
import torch.distributed as dist
'''
local_rank = int(os.environ.get("LOCAL_RANK", 0))

dist.init_process_group(
    backend="nccl",
    init_method="env://",
    rank=int(os.environ["RANK"]),
    world_size=int(os.environ["WORLD_SIZE"]),
    # NB: device_ids is only used by new PyTorch versions if you call barrier with it.
)

torch.cuda.set_device(local_rank)

print(
    f"PID {os.getpid()} | LOCAL_RANK={local_rank} | torch.cuda.current_device()={torch.cuda.current_device()} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}",
    flush=True
)
'''
import math
from time import time
from accelerate import Accelerator, PartialState, init_empty_weights, load_checkpoint_and_dispatch
from accelerate.utils import gather_object, DeepSpeedPlugin
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
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Tuple
from utils.logging_config import LOGGER
from utils.graph_utils import extract_msg_file, clean_data, split_email_thread, chunk_emails

langsmith_api_key = os.environ.get("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
if not langsmith_api_key:
    LOGGER.warning("Langsmith API key not found. Tracing will be disabled.")
    
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

# Constants
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_EMAILS_PER_PROMPT = 4  # Max emails per context batch
MAX_SEQUENCE_LENGTH = 8192  # Llama 3 context limit

plugin = DeepSpeedPlugin(
    hf_ds_config="ds_config.json"  # or a Python dict
)

state = Accelerator(mixed_precision="fp16", deepspeed_plugin=plugin)

rank = state.process_index       # 0 … num_processes–1
world_size = state.num_processes # total number of processes

bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
)

'''
device_map = infer_auto_device_map(
    AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16),
    no_split_module_classes=["LlamaDecoderLayer"],
    dtype=torch.bfloat16
)'''

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
tokenizer.model_max_length = MAX_SEQUENCE_LENGTH

model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            #device_map="auto",  #{"": state.process_index}, 
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
            #attn_implementation="sdpa",
            #offload_folder="offload",  # Directory for CPU offloading
            #max_memory={i: "12GiB" for i in range(8)}  # Reduce to 12GB/GPU
        )

model.config.max_length = tokenizer.model_max_length + 8192
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.eos_token_id

#Stopping criteria for Llama 3 tokens
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [128001, 128009]  # <|end_of_text|> and <|eot_id|>
        return any(stop_id in input_ids[0] for stop_id in stop_ids)
            
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    #device=state.device,  # Ensure HF pipeline sends inputs to correct device
    max_new_tokens = 8192,
    return_full_text=False,
    do_sample=False,
    temperature=None,
    top_k=None,
    top_p=None
    #stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
    #eos_token_id=[tokenizer.eos_token_id, 128009]
)

#model, pipe = state.prepare(model, pipe)
LOGGER.info(f"[Rank {rank}] model+pipeline ready—entering main loop")

llm = HuggingFacePipeline(pipeline=pipe)

SPLIT_EMAILS_CHAIN = (prompt_template | llm | parser)

def split_and_extract_emails_acc(file_path: str) -> List[Dict]:
    """
    Distributed version with parallel invocations using accelerate

        Splits a .msg file containing an email thread into individual cleaned emails, extracts metadata (sender, date, etc.),
    and returns them in chronological order (oldest first). 

    Args:
        file_path (str): Path to the .msg file containing the email thread.
    Returns:
        List[Dict]: A list of dictionaries, each containing the extracted email information.
    """ 


    os.environ["LANGCHAIN_PROJECT"] = "extract_emails_acc"

    raw_msg_content = extract_msg_file(file_path)
    cleaned_msg_content = clean_data(raw_msg_content)
    LOGGER.info("Splitting emails...")
    try:
        splitted_emails = split_email_thread(cleaned_msg_content)[::-1]

        chunks = list(chunk_emails(splitted_emails, 3)) #math.ceil(len(splitted_emails) / state.num_processes)))
        joined_chunks = ["\n***\n".join(chunk) for chunk in chunks]

        results = []
        with state.split_between_processes(joined_chunks) as local_chunks:

            for idx, chunk in enumerate(local_chunks):
                result = SPLIT_EMAILS_CHAIN.invoke({"emails": chunk}, config={"metadata": {"chunk_num": idx, "invocation_method": "distributed inference"}, "run_name": f"chunk_{idx}"})
                results.extend(result)

        #state.wait_for_everyone()
        gloo_group = dist.new_group(backend="gloo")  # Use Gloo for CPU communication

        dist.barrier(backend="gloo")  # Ensure all processes are synchronized before gathering results
        all_results = gather_object(results, group=gloo_group)
        dist.barrier(backend="gloo")
        
        if state.is_main_process:
            flat_results = [item for sublist in all_results for item in sublist]
            LOGGER.info(f"Total output emails: {len(flat_results)}")
            return flat_results
        else:
            return []
        '''
        state.wait_for_everyone()
        return results'''
    
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

    if not os.path.isdir(dir_path):
        LOGGER.error(f"{dir_path} is not a valid directory.")
        sys.exit(1)

    folder_name = os.path.basename(os.path.normpath(dir_path))

    output_path = os.path.join(dir_path, f"{folder_name}.json")

    email_data = []
    #graph = nx.DiGraph()
    for filename in os.listdir(dir_path):
        if filename.endswith(".msg"):
            file_path = os.path.join(dir_path, filename)
            try:

                tic = time()
                
                local_data = split_and_extract_emails_acc(file_path)
                LOGGER.info(f"[Rank {rank}] Processed {filename} in {time() - tic:.2f}s")
                email_data.extend(local_data) 
                '''
                # Write *only this rank's* results for that file
                out_fname = os.path.join(dir_path, f"{folder_name}_rank{rank}.json")
                with open(out_fname, "w", encoding="utf-8") as f:
                    json.dump(local_data, f, indent=4, ensure_ascii=False)

            except Exception as e:
                LOGGER.error(f"[Rank {rank}] Error processing {filename}: {e}")

        # 2) Barrier so every rank has finished writing its partial files
        state.wait_for_everyone()

        # 3) Only the main rank reads & merges them
        if state.is_main_process:
            combined = []
            for r in range(world_size):
                partial_path = os.path.join(dir_path, f"{folder_name}_rank{r}.json")
                if os.path.exists(partial_path):
                    with open(partial_path, "r", encoding="utf-8") as f:
                        combined.extend(json.load(f))

            # 4) Write the final merged output
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(combined, f, indent=4, ensure_ascii=False)
        '''
            except Exception as e:
                LOGGER.error(f"Error processing {filename}: {e}")
            #LOGGER.info(f"[Rank 0] Wrote combined output to {output_path}")

    
    
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(email_data, file, indent=4, ensure_ascii=False, default=str)
    