
import threading, os, json
from typing import Any, Dict, Optional

STATUS_FILE = "job_status_db.json"
_lock = threading.Lock()

def _load_status() -> Dict[str, Dict[str, Any]]:
    """Loads all job status records from the persistent file."""
    if not os.path.exists(STATUS_FILE):
        return {}
    
    with _lock:
        try:
            with open(STATUS_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
        
def _save_status(data: Dict[str, Dict[str, Any]]):
    """Saves all job status records to the persistent file."""
    with _lock:
        with open(STATUS_FILE, 'w') as f:
            json.dump(data, f, indent=4)

def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves the status for a specific job ID."""
    data = _load_status()
    return data.get(job_id)

def update_job_status(job_id: str, updates: Dict[str, Any]):
    """Updates the status for a job ID in a thread-safe manner."""
    data = _load_status()
    if job_id not in data:
        data[job_id] = {}
        
    data[job_id].update(updates)
    _save_status(data)

# Initialize file if it doesn't exist
if not os.path.exists(STATUS_FILE):
    _save_status({})