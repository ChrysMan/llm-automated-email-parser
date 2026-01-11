import math
from ..utils.logging import LOGGER
from typing import List, Tuple

def find_best_chunk_size(total_emails, min_chunk, max_chunk):
    best_chunk = min_chunk
    highest_remainder = -1  

    for chunk_size in range(max_chunk, min_chunk - 1, -1):
        remainder = total_emails % chunk_size
        if remainder == 0:
            return chunk_size  # Prefer exact division first
        elif remainder > highest_remainder:
            highest_remainder = remainder
            best_chunk = chunk_size

    return best_chunk  # Return the chunk size with the highest leftover

def chunk_emails(email_list, chunk_size):
    """ Yield successive chunks of n emails from the list. """
    for i in range(0, len(email_list), chunk_size):
        yield email_list[i:i + chunk_size]


def split_for_gpus_dynamic(
    emails: List[str],
    num_gpus_available: int,
    min_per_chunk: int,
    max_per_chunk: int
) -> Tuple[List[List[List[str]]], int]:
    """
   Returns (batch_of_chunks, gpus_used)

    batch_of_chunks[i]  ->  List[List[str]]  (all chunks assigned to GPU i)
    gpus_used           ->  int  (how many GPUs you should launch/use)

    Guarantees:
    • every chunk length <= max_per_chunk
    • chunk sizes differ by at most 1
    • GPUs are used evenly (±1 chunk difference)
    • if len(emails) <= max_per_chunk -> one GPU, one chunk
    """
    n = len(emails)
    if n == 0:
        return [], 0

    # ---------- single GPU ----------
    if n <= max_per_chunk:
        return [[[e for e in emails]]], 1     # one GPU, one chunk

    # ---------- how many GPUs do we actually need? ----------
    # At least enough so that one chunk per GPU fits under the max size
    gpus_used = min(num_gpus_available, math.ceil(n / max_per_chunk))

    # ---------- decide how many chunks ----------
    mid_size      = (min_per_chunk + max_per_chunk) / 2          
    ideal_chunks  = math.ceil(n / mid_size)
    chunk_count   = max(ideal_chunks, gpus_used)   # at least one chunk per GPU

    # ---------- continuous slicing into `chunk_count` ----------
    base, rem = divmod(n, chunk_count)
    chunks: List[List[str]] = [] 
    idx =  0
    for i in range(chunk_count):
        next_idx = idx + base + (1 if i < rem else 0)
        chunks.append(emails[idx:next_idx])
        idx = next_idx

    # ---------- evenly distribute chunks across GPUs ----------
    base_cpg, extra = divmod(len(chunks), gpus_used)
    batch_of_chunks: List[List[List[str]]] = []
    cursor = 0
    for i in range(gpus_used):
        chunk_size = base_cpg + (1 if i < extra else 0)
        batch_of_chunks.append(chunks[cursor:cursor + chunk_size])
        cursor += chunk_size

    return batch_of_chunks, gpus_used


def smart_chunker(
    emails: List[str],
    num_gpus_available: int,
    min_size: int,
    max_size: int
) -> Tuple[List[List[str]], int]:
    """
   Returns (chunks, gpus_used)

    chunks[]            ->  List[List[str]]  (all chunks assigned to GPU i)
    gpus_used           ->  int  (how many GPUs you should launch/use)

    Guarantees:
    • every chunk length <= max_per_chunk
    • chunk sizes differ by at most 1
    • GPUs are used evenly (±1 chunk difference)
    • if len(emails) <= max_per_chunk -> one GPU, one chunk
    """
    n = len(emails)
    if n == 0:
        return [], 0

    # ---------- single GPU ----------
    if n <= max_size:
        return [emails], 1     # one GPU, one chunk

    # ---------- how many GPUs do we actually need? ----------
    # At least enough so that one chunk per GPU fits under the max size
    gpus_used = min(num_gpus_available, math.ceil(n / max_size))

    # ---------- decide how many chunks ----------
    mid_size      = (min_size + max_size) / 2          
    ideal_chunks  = math.ceil(n / mid_size)
    chunk_count   = max(ideal_chunks, gpus_used)   # at least one chunk per GPU

    # ---------- continuous slicing into `chunk_count` ----------
    base, rem = divmod(n, chunk_count)
    chunks: List[List[str]] = [] 
    idx =  0
    for i in range(chunk_count):
        next_idx = idx + base + (1 if i < rem else 0)
        chunks.append(emails[idx:next_idx])
        idx = next_idx

    return chunks, gpus_used