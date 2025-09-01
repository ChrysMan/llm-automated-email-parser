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
from langchain_core.exceptions import OutputParserException
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from utils.logging_config import LOGGER
from utils.graph_utils import split_for_gpus_dynamic, extract_msg_file, append_file, chunk_emails, clean_data, split_email_thread
from dotenv import load_dotenv

assert Version(ray.__version__) >= Version(
    "2.22.0"), "Ray version must be at least 2.22.0"

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
client = Client()

#os.environ["LANGCHAIN_PROJECT"] = "extract_emails_vllm"
if langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
else:
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
2. In the case were the email thread contains only one email, do not split the email thread.
3. Return the unique emails **ONLY** as a JSON list, where each email is a **separate object** with the following fields:

- sender: The sender of the email. Store their name and email, if available, as a string in the format "Name <email@example.com>". If only a name is present, store it as "Name". 
- sent: Extract the date and write it in this format: Full weekday name, full month name day, four-digit year, hour:minute:second AM/PM. 
- to: A list of recipients' names and emails, if available. Store each entry as a string in the format "Name <email@example.com>" or "Name". The "To" field may contain multiple recipients separated by commas or semicolons but is usually one.
- cc: A list of additional recipients' names and emails. Store each entry as a string in the format "Name <email@example.com>" or "Name". The "Cc" field may contain multiple recipients separated by commas or semicolons.
- subject: The subject of the email, stored as a string. Extract this after the "Subject:" field.
- body:  Include the full content of the message starting from the line **after** "Subject:" or "wrote:", and continue **until the next delimiter** (if it exists). Do not summarize or skip any content.

4. Maintain the chronological order of the emails in the output and be really careful to not change the dates.
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

    def __call__(self, batch_of_chunks: List[List[str]]) -> List[List[Dict]]:
        """Generates structured outputs from batches of email chunks."""
        
        prompts = [
            prompt_template.format(emails="\n***\n".join(chunks))
            for chunks in batch_of_chunks
        ]

        with trace(
            name=f"actor_{ray.get_runtime_context().get_actor_id()}",
            project_name="extract_emails_vllm",
            inputs={"prompt": prompts},
            metadata={
                "num_of_chunks": len(batch_of_chunks),
                "invocation_method": "distributed inference with vllm"
            },
        )as run:  
            try:                                  
                outputs_obj = self.llm.generate(prompts, self.sampling_params)
                raw = [o.outputs[0].text for o in outputs_obj]
                parsed = [self.parser.parse(r) for r in raw]
                run.outputs={
                    "raw_outputs": raw,
                    "parsed_json": parsed
                }        
            except Exception as e:
                run.error = f"{type(e).__name__}: {e}"
        
        parsed_batches: List[Dict] = []
        
        try:
            parsed_batches.append(parsed) #List[str] = self.parser.parse(raw)
        except OutputParserException as e:
            LOGGER.error(f"JSON parsing failed for chunk of size {len(batch_of_chunks)}: {e}")
            parsed_batches.append([]) 

        return parsed_batches

ray.init(ignore_reinit_error=True)

@ray.remote(num_gpus=1, num_cpus=1)
class RayLLMActor:
    def __init__(self, model_tag: str, tensor_parallel_size: int):
        self.predictor = LLMPredictor(model_tag, tensor_parallel_size)

    def warmup(self):
        """Load the model and compile kernels once."""      
        _ = self.predictor.llm.generate(["warm-up"], SamplingParams(max_tokens=1))
        return "ready"
    
    def predict(self, chunk_batch_prompts: List[List[str]]) -> List[List[Dict]]:       
        return self.predictor(chunk_batch_prompts)


def split_emails(file_path: str, actors: List[RayLLMActor]) -> List[Dict]:

    raw_msg_content = extract_msg_file(file_path)
    cleaned_msg_content = clean_data(raw_msg_content)
    append_file(cleaned_msg_content, "clean.txt")
    LOGGER.info("Splitting emails...")

    gathered_emails: List[Dict] = []

    try:

        #  Implement logic for larger number of emails.
        splitted_emails: List[str] = split_email_thread(cleaned_msg_content)[::-1]
        LOGGER.info(f"Splitted emails: {len(splitted_emails)}")
        batch_of_chunks, gpus_needed = split_for_gpus_dynamic(splitted_emails, num_instances, min_per_chunk= 4, max_per_chunk=7)

        actors_to_use = actors[:gpus_needed]

        pool = ActorPool(actors_to_use)
        batch_results = list(pool.map(lambda a, c: a.predict.remote(c), batch_of_chunks))

        gathered_emails = [d for batch in batch_results for sublist in batch for d in sublist]

        LOGGER.info("Splitted and extracted emails...")      

        return gathered_emails  
    
    except Exception as e:
        LOGGER.error(f"Failed to split emails: {e}")


    return gathered_emails


if __name__ == "__main__":
    if len(sys.argv) != 2:
        LOGGER.error("Usage: python emailParsing.py <dir_path>")
        sys.exit(1)
    
    dir_path = sys.argv[1]

    if not os.path.isdir(dir_path):
        LOGGER.error(f"{dir_path} is not a valid directory.")
        sys.exit(1)

    folder_name = os.path.basename(os.path.normpath(dir_path))

    output_path = os.path.join(dir_path, f"{folder_name}.json")

    # Create 8 actors, each pinned to 1 gpu 
    model_tag = "meta-llama/Llama-3.1-8B-Instruct"
    actors = [RayLLMActor.remote(model_tag, tensor_parallel_size) for _ in range(num_instances)]
    
    ray.get([a.warmup.remote() for a in actors])

    email_data = []
    graph = nx.DiGraph()


    try: 

        for filename in os.listdir(dir_path):
            if filename.endswith(".msg"):
                file_path = os.path.abspath(os.path.join(dir_path, filename))
                try:               
                    result = split_emails(file_path, actors)
                    #graph = add_to_graph(graph, result, filename)
                    email_data.extend(result)
                except Exception as e:
                    LOGGER.error(f"Processing {filename} failed: {e}")
        
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(email_data, file, indent=4, ensure_ascii=False, default=str)
   
    except Exception as main_exc:
        LOGGER.error(f"Fatal error in main: {main_exc}")

    finally:
        client.flush()
        ray.shutdown()
    

    
