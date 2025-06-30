import json, os, sys
import networkx as nx
import ray
import math
from ray.util.actor_pool import ActorPool
from packaging.version import Version
from vllm import LLM, SamplingParams
from langsmith import trace, Client
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from utils.logging_config import LOGGER
from utils.graph_utils import smart_chunker, extract_msg_file, append_file, clean_data, split_email_thread

assert Version(ray.__version__) >= Version(
    "2.22.0"), "Ray version must be at least 2.22.0"

langsmith_api_key = os.environ.get("LANGSMITH_API_KEY")

client = Client()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
#os.environ["LANGCHAIN_PROJECT"] = "extract_emails_vllm"
if not langsmith_api_key:
    LOGGER.warning("Langsmith API key not found. Tracing will be disabled.")
    

# Set tensor parallelism per instance.
tensor_parallel_size = 1

# Set number of instances. Each instance will use tensor_parallel_size GPUs.
num_instances = 8

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

class LLMPredictor:

    def __init__(self, model_tag:str, tensor_parallel_size: int):
        # Create an LLM.
        self.llm = LLM(model=model_tag,
                       tensor_parallel_size=tensor_parallel_size, 
                       dtype = 'float16',
                       gpu_memory_utilization=0.95,
                       max_seq_len_to_capture=4096,
                       max_model_len=8192,
                       cpu_offload_gb=3,
                       enforce_eager=True
            )
        
        self.parser = parser
        self.sampling_params = SamplingParams(temperature=0, max_tokens=3500)

    def __call__(self, chunk: List[str]) -> List[Dict]:
        """Generates structured outputs from email chunk."""
        
        prompts = prompt_template.format(emails="\n***\n".join(chunk))
            
        with trace(
            name=f"actor_{ray.get_runtime_context().get_actor_id()}",
            project_name="extract_emails_vllm",
            inputs={"prompt": prompts},
            metadata={
                "num_of_chunks": len(chunk),
                "invocation_method": "distributed inference with vllm"
            },
        )as run:  
            try:                                  
                outputs_obj = self.llm.generate(prompts, self.sampling_params)
                raw = outputs_obj.outputs[0].text 
                parsed = self.parser.parse(raw) 
                run.outputs={
                    "raw_outputs": raw,
                    "parsed_json": parsed
                }        
                return parsed  
            except Exception as e:
                run.error = f"{type(e).__name__}: {e}"
                LOGGER.error(f"JSON parsing failed for chunk of size {len(chunk)}: {e}")
                return []
        

ray.init(ignore_reinit_error=True)

@ray.remote(num_gpus=1, num_cpus=1)
class RayLLMActor:
    def __init__(self, model_tag: str, tensor_parallel_size: int):
        self.predictor = LLMPredictor(model_tag, tensor_parallel_size)

    def warmup(self):
        """Load the model and compile kernels once."""      
        _ = self.predictor.llm.generate(["warm-up"], SamplingParams(max_tokens=1))
        return "ready"
    
    def predict(self, chunk: List[str]) -> List[Dict]:       
        return self.predictor(chunk)

def process_directory_distributed(dir_path: str, actors: List[RayLLMActor]) -> List[Dict]:
    """
    Processes all .msg files in a directory using dynamic task assignment across actors.
    """
    indexed_chunks = []
    filenames = []
    global_index = 0

    for filename in os.listdir(dir_path):
        if not filename.endswith(".msg"):
            continue
        
        file_path = os.path.join(dir_path, filename)
        raw_msg_content = extract_msg_file(file_path)
        cleaned_msg_content = clean_data(raw_msg_content)
        append_file(cleaned_msg_content, "clean.txt")

        try:
            splitted_emails: List[str] = split_email_thread(cleaned_msg_content)[::-1]
            chunks, gpus_needed = smart_chunker(splitted_emails, num_instances, min_size=4, max_size=6)
            for chunk in chunks:
                indexed_chunks.append((global_index, chunk))
                filenames.append(filename)
                global_index += 1
        except Exception as e:
            LOGGER.error(f"Failed to preprocess {filename}: {e}")

    actors_to_use = actors[:gpus_needed]
    pool = ActorPool(actors_to_use)
    ordered_results = [None] * len(indexed_chunks)

    for idx, chunk in indexed_chunks:
        pool.submit(lambda a, c: (c[0], a.predict.remote(c[1])), (idx, chunk))

    # Gather and reorder
    for _ in range(len(indexed_chunks)):
        idx, parsed = pool.get_next_unordered()
        ordered_results[idx] = parsed

        return ordered_results


if __name__ == "__main__":
    if len(sys.argv) != 2:
        LOGGER.error("Usage: python emailParsing.py <dir_path>")
        sys.exit(1)

    dir_path = sys.argv[1]
    if not os.path.isdir(dir_path):
        LOGGER.error(f"{dir_path} is not a valid directory.")
        sys.exit(1)

    model_tag = "meta-llama/Llama-3.1-8B-Instruct"
    actors = [RayLLMActor.remote(model_tag, tensor_parallel_size=1) for _ in range(num_instances)]
    ray.get([a.warmup.remote() for a in actors])

    try:
        email_data = process_directory_distributed(
            dir_path=dir_path,
            actors=actors
        )

        output_path = os.path.join(dir_path, f"{os.path.basename(dir_path)}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(email_data, f, indent=4, ensure_ascii=False, default=str)

    except Exception as e:
        LOGGER.error(f"Fatal error in main: {e}")

    finally:
        client.flush()  # Ensure all traces are sent to LangSmith
        ray.shutdown()

    

    
