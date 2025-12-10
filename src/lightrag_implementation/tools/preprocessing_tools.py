import os, json, re, uuid, asyncio
from time import time
from multiprocessing import Process
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from typing import Any, Dict
from src.email_preprocessing_vllm_serve import LLMPredictor
from src.deduplicate_cleaned_emails import deduplicate_emails
from src.utils.prompts import cleaning_prompt,formatter_and_translator_prompt
from src.utils.graph_utils import extract_msg_file, clean_data, split_email_thread
from src.utils.logging_config import LOGGER
from . import job_persistence 

class PreprocessingInput(BaseModel):
    dir_path: str = Field(description="The directory path containing the .msg email files to process.")

def _long_running_preprocessing_worker(job_id: str, dir_path: str, output_path: str):
    """The function that runs in a separate process/thread."""
    try:
        tic = time.time()
        job_persistence.update_job_status(job_id, {"status": "RUNNING", "output_path": output_path})

        predictor = LLMPredictor()

        all_emails_to_process = []
        for filename in os.listdir(dir_path):
            if filename.endswith(".msg"):
                file_path = os.path.join(dir_path, filename)

                try:
                    raw_msg_content = extract_msg_file(file_path)
                    cleaned_msg_content = clean_data(raw_msg_content)
                    all_emails_to_process.extend(split_email_thread(cleaned_msg_content))

                except Exception as e:
                    LOGGER.error(f"Failed to extract or clean email from {filename}: {e}")
                    continue

        # Prepare all prompts outside the file loop
        formatting_prompts = [formatter_and_translator_prompt.format(email=e) for e in all_emails_to_process]
        results = predictor(formatting_prompts)

        str_results = [str(r) for r in results]

        cleaning_prompts = [cleaning_prompt.format(email=e) for e in str_results]
        results = predictor(cleaning_prompts)

        # ---------------Deduplicate results---------------
        unique_emails = deduplicate_emails(results)
        emails_json = []
        for text in unique_emails:
                # Split at the first "body:" (case-insensitive, multi-line safe)
                parts = re.split(r'(?mi)^\s*body\s*:\s*', text, maxsplit=1)
                headers_part = parts[0]
                body_part = parts[1] if len(parts) == 2 else ""

                email_dict = {}
                for line in headers_part.splitlines():
                    if ":" in line:
                        key, val = line.split(":", 1)  # split only on first colon
                        email_dict[key.strip().lower()] = val.strip()

                email_dict["body"] = body_part.rstrip()
                emails_json.append(email_dict)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(emails_json, file, indent=4, ensure_ascii=False, default=str)
        
        toc = time.time()
        job_persistence.update_job_status(job_id, {
            "status": "COMPLETE", 
            "duration": round(toc - tic, 2)
        })

    except Exception as e:
        job_persistence.update_job_status(job_id, {
            "status": "FAILED", 
            "error": str(e)
        })

class ExecutePreprocessingTool(BaseTool):
    name = "execute_full_preprocessing"
    description = (
        "Starts a complex, long-running background job (100-500s) to extract, clean, "
        "translate, and deduplicate emails in the specified directory. Returns the job ID and initial status immediately. "
        "MUST be followed by calls to a polling tool."
    )
    args_schema = PreprocessingInput

    def _run(self, dir_path: str) -> Dict[str, str]:
        if not os.path.isdir(dir_path):
            return {"status": "FAILED", "reason": f"{dir_path} is not a valid directory."}
        
        job_id = str(uuid.uuid4())
        
        folder_name = os.path.basename(os.path.normpath(dir_path))
        
        output_path = os.path.join(dir_path, f"{folder_name}_unique.json")

        # Write the initial RUNNING status to the persistent file
        job_persistence.update_job_status(job_id, {"status": "LAUNCHING", "output_path": output_path})
        
        # Start the worker process
        p = Process(target=_long_running_preprocessing_worker, args=(job_id, dir_path, output_path))
        p.start()
        
        # Return the ID immediately
        return {
            "status": "JOB_STARTED", 
            "job_id": job_id,
            "message": f"Preprocessing job {job_id} launched successfully."
        }

    async def _arun(self, dir_path: str) -> Dict[str, str]:
        return await super()._arun(dir_path)

class CheckStatusInput(BaseModel):
    job_id: str = Field(description="The unique ID of the long-running preprocessing job.")


class CheckPreprocessingStatusTool(BaseTool):
    name = "check_preprocessing_status"
    description = "Checks the current status and output path of a background preprocessing job using its job ID."
    args_schema = CheckStatusInput

    def _run(self, job_id: str) -> Dict[str, Any]:
        """Synchronous check against the persistent store."""
        from . import job_persistence
        
        status_data = job_persistence.get_job_status(job_id)
        
        if status_data:
            # Return the full status dict for the agent to parse
            return status_data
        else:
            return {"status": "UNKNOWN", "message": f"Job ID {job_id} not found."}

    async def _arun(self, job_id: str) -> Dict[str, Any]:
        # Simple I/O operation, can wrap in asyncio.to_thread for async compliance
        return await asyncio.to_thread(self._run, job_id)