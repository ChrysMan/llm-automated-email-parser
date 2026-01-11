import os
import json
from ..utils.logging import LOGGER
from typing import List

def write_file(content, filename):
    """ Writes the given content to the given file to the local directory """
    try:
        with open(f"{filename}", "w", encoding="utf-8") as f:
            f.write(str(content))
    except Exception as e:
        LOGGER.error(f"Failed to write at file {filename}: {e}")

def append_file(content, filename):
    """ Appends the given content to given file to the local directory """
    try:
        with open(f"{filename}", "a", encoding="utf-8") as f:
            f.write(str(content))
    except Exception as e:
        LOGGER.error(f"Failed to append at file {filename}: {e}")

def read_json_file(filename:str) -> List[dict]:
    """ Reads a JSON file and returns its content as a list of dictionaries """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        LOGGER.error(f"Failed to read JSON file {filename}: {e}")
        return []
    
def find_file(filename, search_path):
        for root, dirs, files in os.walk(search_path):
            if filename in files:
                return os.path.join(root, filename)
        return None

def find_dir(dirname, search_path):
    for root, dirs, files in os.walk(search_path):
        if dirname in dirs:
            return os.path.join(root, dirname)
    return None