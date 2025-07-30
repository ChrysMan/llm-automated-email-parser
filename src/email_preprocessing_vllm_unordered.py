import json, os, sys
import ray
import traceback
from time import time
from ray.util.actor_pool import ActorPool
from packaging.version import Version
from vllm import LLM, SamplingParams
from langsmith import trace, Client
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import Optional, List, Dict, Tuple
from pydantic import BaseModel, Field
from utils.logging_config import LOGGER
from utils.graph_utils import smart_chunker, extract_msg_file, clean_data, split_email_thread, write_file, append_file

assert Version(ray.__version__) >= Version(
    "2.22.0"), "Ray version must be at least 2.22.0"

langsmith_api_key = os.environ.get("LANGSMITH_API_KEY")

client = Client()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
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
"""<|start_header_id|>system<|end_header_id|>
You are an expert assistant that extracts structured information from email threads with no loss of content.

1. Split the input into individual emails:
- Use "-***-" as a delimiter when present.
- If missing, split based on headers like "From:", "Sent:", "To:", "Cc:", "Subject:", which may appear in other languages. Translate these headers to English before splitting.

2. Translate only the **header labels** to English. Do not change dates, times, or names. If month or weekday names are in another language, translate them to English without changing the date/time values.

3. Return a JSON array where each email is an object with:
- `sender`: Extract from "From:". Format as "Name <email@example.com>" or "Name".
- `sent`: Extract from "Sent:". Keep the exact date/time but translate month/weekday names to English. Example: "Friday, March 22, 2024 05:19:00 PM".
- `to`: List of recipients from "To:", formatted as "Name <email@example.com>" or "Name".
- `cc`: Same as `to`, from "Cc:".
- `subject`: Extract from "Subject:".
- `body`: All text up to the next header or "-***-". Do not include "-***-" in the body. If empty, return an empty string.

4. Maintain the order of emails as they appear.  
5. Remove signatures, trackers, footers, metadata, and any other non-essential content from the body. Focus on the main content of the email. 
5. Do not hallucinate or add any information that is not present in the email thread. If you are unsure about a date, copy it exactly as shown, translating only the weekday/month names.
6. The length of the JSON list needs to be the same as the number of the emails strictly.
7. Output only the **raw JSON array**, with no explanations or extra text.

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
        
        prompts = prompt_template.format(emails="\n-***-\n".join(chunk))
            
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
                raw = outputs_obj[0].outputs[0].text
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
            finally:
                run.end()        

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
    
    def predict_with_index(self, idx_and_chunk: Tuple[int, List[str]]) -> Tuple[int, List[Dict]]:
        idx, chunk = idx_and_chunk
        result = self.predictor(chunk)
        return idx, result

def process_directory_distributed(dir_path: str) -> List[Dict]:
    """
    Processes all .msg files in a directory using dynamic task assignment across actors.
    Returns a flat list of extracted email dictionaries.
    """
    indexed_chunks = []
    filenames = []
    cleaned_msg_content = []
    global_index = 0

     # --- Load and clean all .msg files ---
    for filename in os.listdir(dir_path):
        if not filename.endswith(".msg"):
            continue
        
        file_path = os.path.join(dir_path, filename)

        try:
            raw_msg_content = extract_msg_file(file_path)
            append_file(raw_msg_content, "Raw.txt")
            cleaned_msg_content.append(clean_data(raw_msg_content))
            append_file(cleaned_msg_content[-1], "Cleaned.txt")  
        except Exception as e:
            LOGGER.error(f"Failed to process {filename}: {e}")
            return []

    # --- Split and chunk emails ---
    try:
        final_msg_content = "\n\n".join(cleaned_msg_content)
        splitted_emails: List[str] = split_email_thread(final_msg_content)[::-1]
        chunks, gpus_needed = smart_chunker(splitted_emails, num_instances, min_size=4, max_size=6)
        LOGGER.info(f"Total emails split: {len(splitted_emails)}, Chunks created: {len(chunks)}, GPUs needed: {gpus_needed}")

        for chunk in chunks:
            indexed_chunks.append((global_index, chunk))
            filenames.append(filename)
            global_index += 1
    except Exception as e:
        LOGGER.error(f"Failed to preprocess {filename}: {e}")

    # --- Start the actors dynamically based on GPUs needed ---
    actors = [RayLLMActor.remote(model_tag, tensor_parallel_size) for _ in range(gpus_needed)]
    ray.get([a.warmup.remote() for a in actors])
    pool = ActorPool(actors)

    # --- Submit tasks and collect results ---
    try:
        ordered_results: List[Optional[List[Dict]]] = [None] * len(indexed_chunks)

        for idx, chunk in indexed_chunks:
            pool.submit(lambda a, c: a.predict_with_index.remote(c), (idx, chunk))

        for _ in range(len(indexed_chunks)):
            try:
                idx, parsed = pool.get_next_unordered()
                ordered_results[idx] = parsed
            except Exception as e:
                LOGGER.error(f"Error retrieving result from actor pool: {e}")

        flattened_results = [email for batch in ordered_results if batch for email in batch]

        return flattened_results

    finally:
        for actor in actors:
            try:
                ray.kill(actor)
            except Exception as e:
                LOGGER.warning(f"Error killing actor: {e}")


if __name__ == "__main__":

    tic = time()

    if len(sys.argv) != 2:
        LOGGER.error("Usage: python emailParsing.py <dir_path>")
        sys.exit(1)

    dir_path = sys.argv[1]
    if not os.path.isdir(dir_path):
        LOGGER.error(f"{dir_path} is not a valid directory.")
        sys.exit(1)

    #model_tag = "meta-llama/Llama-3.1-8B-Instruct"
    model_tag = "Qwen/Qwen2.5-7B-Instruct"

    try:
        email_data = process_directory_distributed(dir_path=dir_path)

        output_path = os.path.join(dir_path, f"{os.path.basename(dir_path)}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(email_data, f, indent=4, ensure_ascii=False, default=str)

        LOGGER.info(f"Time taken to process: {time() - tic} seconds")

    except Exception as e:
        LOGGER.error("Fatal error in main:\n%s", traceback.format_exc())

    finally:
        client.flush()  # Ensure all traces are sent to LangSmith
        ray.shutdown()  # Clean up actors and Ray runtime

    

    
