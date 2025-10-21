# Automating Information Extraction from Emails using Large Language Models

*This project is still in progress* 

## Prerequisites
- Python 3.12 installed
- `ollama` installed ([Installation Guide](https://ollama.ai/))
- `pip` installed (comes with Python)
- `.msg` email files available for processing
- LangSmith api key 
- Huggingface api key




## Setup & Execution:

### Create a Virtual Environment:
Run the following command in your project's root directory:

#### For Linux/macOS: 
```sh
python3 -m venv myvenv  
```
or
```sh
conda create -n myvenv python=3.12 -y
```
#### For Windows: 
```sh
python -m venv myvenv   
```
or
```sh
conda create -n myvenv python=3.12 -y
```

### Activate the Virtual Environment:
#### For Linux/macOS: 
```sh
source myvenv/bin/activate 
```
or
```sh
conda activate myvenv
```
#### For Windows (Command Prompt): 
```sh
venv\Scripts\activate 
```
or
```sh
conda activate myvenv
```
### Install Dependencies:
```sh
pip install -r requirements.txt
```

### Download the LLM Model
Before running the program, download the required model (or any model you wish to use):
```
ollama pull llama3.1
```
### Execute program:

#### Different approaches for preprocessing the emails 
```sh
# Synchronous model inference (serial execution)
python email_preprocessing_sync.py /path/to/your/data/directory   
``` 
```sh
# Asynchronous model inference                                
python email_preprocessing_async.py /path/to/your/data/directory  
```
```sh        
# Distributed model inference using vllm                         
python email_preprocessing_vllm_unordered.py /path/to/your/data/directory    
```
```sh
# Distributed model inference using accelerate with distinct prompts 
# (will be transformed into an agent later)
python email_preprocessing_agent /path/to/your/data/directory
```
#### First approach: RAG on Vector Database
```sh
# Create a vector database from a deduplicated email list
python create_vectorDB_spaCyfaiss.py /path/to/your/data/file
# Run Retrieval-Augmented Generation (RAG) on the vector database
python rag_embedDB.py
```
#### Second approach: RAG on Knowledge Graph
```sh
# Create Neo4j knowledge graph from emails
python create_kg.py /path/to/your/data/directory
# Run the bot with an agent implementing RAG on the knowledge graph
python bot.py  
```
#### Note: 
/path/to/your/data/directory should be replaced with the actual path where your `.msg` files are stored.
