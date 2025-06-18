import json, os, sys
import networkx as nx
import ray
import math
from packaging.version import Version
from vllm import LLM, SamplingParams
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from utils.logging_config import LOGGER
from utils.graph_utils import extract_msg_file, append_file, chunk_emails, clean_data, split_email_thread

assert Version(ray.__version__) >= Version(
    "2.22.0"), "Ray version must be at least 2.22.0"

langsmith_api_key = os.environ.get("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
if not langsmith_api_key:
    LOGGER.warning("Langsmith API key not found. Tracing will be disabled.")
    

sampling_params = SamplingParams(temperature=0, max_tokens=8192)

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
                       gpu_memory_utilization=0.8,
                       max_seq_len_to_capture=4096,
                       max_model_len=8192,
                       cpu_offload_gb=6,
                       enforce_eager=True
            )
        
        self.parser = parser

        #self.SPLIT_EMAILS_CHAIN = (prompt_template | unwrap_vllm | parser)

    def __call__(self, batch: List[str]) -> List[Dict]:
        ''' Generate texts from the prompts.
         The output is a list of RequestOutput objects that contain the prompt,
         generated text, and other information.'''
        
        # One big "joined prompt" into the chain to retain information between emails.
        joined_chunks = ["\n***\n".join(batch)]
        outputs = self.llm.generate(joined_chunks, sampling_params)
        raw = outputs[0].outputs[0].text
        try:
            generated_text: List[Dict] = self.parser.parse(raw)
        except OutputParserException as e:
            LOGGER.error(f"JSON parsing failed for chunk of size {len(batch)}: {e}")
            raise

        #generated_text = self.SPLIT_EMAILS_CHAIN.invoke({"emails": joined_chunks}, config={"metadata": {"chunk_num": len(batch), "invocation_method": "distributed inference with vllm"}, "run_name": f"actor_{ray.get_runtime_context().get_actor_id()}"})
        
        '''
        # Batch mode with a list of emails
        outputs = self.llm.generate(batch["text"], sampling_params)
        prompt: List[str] = []
        generated_text: List[str] = []
        for output in outputs:
            parsed_output = parser.parse(output.outputs[0].text)
            generated_text.append(' '.join([o.text for o in output.outputs]))
        '''
        return generated_text

ray.init(ignore_reinit_error=True)

@ray.remote(num_gpus=1, num_cpus=1)
class RayLLMActor:
    def __init__(self, model_tag: str, tensor_parallel_size: int):
        self.predictor = LLMPredictor(model_tag, tensor_parallel_size)

    def predict(self, batch_prompts: List[str]) -> List[Dict]:       
        return self.predictor(batch_prompts)


def split_emails(file_path: str, actors: RayLLMActor) -> List[Dict]:

    os.environ["LANGCHAIN_PROJECT"] = "extract_emails_vllm"

    raw_msg_content = extract_msg_file(file_path)
    cleaned_msg_content = clean_data(raw_msg_content)
    append_file(cleaned_msg_content, "clean.txt")
    LOGGER.info("Splitting emails...")

    gathered_emails: List[Dict] = []

    try:
        splitted_emails: List[str] = split_email_thread(cleaned_msg_content)[::-1]
        chunk_size= math.ceil(len(splitted_emails) / num_instances)
        chunks: List[List[str]] = list((chunk_emails(splitted_emails, chunk_size=chunk_size)))

        futures = []
        for idx, batch in enumerate(chunks):
            # Create a batch of prompts for the LLM
            actor = actors[idx % num_instances]
            futures.append(actor.predict.remote(batch))

        batch_results: List[List[Dict]] = ray.get(futures)

        for sublist in batch_results:
            gathered_emails.extend(sublist)
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

    # Create 8 actros, each pinned to 1 gpu 
    model_tag = "meta-llama/Llama-3.1-8B-Instruct"
    actors = [RayLLMActor.remote(model_tag, tensor_parallel_size) for _ in range(num_instances)]
    
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
        ray.shutdown()
    
'''
    plt.figure(figsize=(12, 10))
    pos = nx.kamada_kawai_layout(graph)
    node_options = {"node_color": "black", "node_size": 10}
    edge_options = {"width": 1, "alpha": 0.5, "edge_color": "black", "arrowsize": 5, "connectionstyle": 'arc3,rad=0.2'}
    label_options = {"font_size": 5, "font_color": "blue", "verticalalignment": "top", "horizontalalignment": "right"}
    nx.draw_networkx_nodes(graph, pos, **node_options)
    nx.draw_networkx_edges(graph, pos, **edge_options)
    nx.draw_networkx_labels(graph, pos, **label_options)
    plt.savefig("graph.png")
'''
    
