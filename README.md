# Automating Information Extraction from Emails using Large Language Models

*This project is still in progress* 

## Prerequisites
- Python 3.12 installed
- `ollama` installed ([Installation Guide](https://ollama.ai/))
- `pip` installed (comes with Python)
- `.msg` email files available for processing
- LangSmith api key 




## Setup & Execution:

### Create a Virtual Environment:
Run the following command in your project's root directory:

#### For Linux/macOS: 
```sh
python3 -m venv venv
```
#### For Windows: 
```sh
python -m venv venv
```

### Activate the Virtual Environment:
#### For Linux/macOS: 
```sh
source venv/bin/activate
```
#### For Windows (Command Prompt): 
```sh
venv\Scripts\activate
```
### Install Dependencies:
```sh
pip install -r requirements.txt
```

### Download the LLM Model
Before running the program, download the required model:
```
ollama pull llama3.1
```
### Execute program:
```sh
 # Synchronous model inference (serial execution)
python email_preprocessing_sync.py /path/to/your/data/directory    
# Asynchronous model inference                                
python email_preprocessing_async.py /path/to/your/data/directory          
# Distributed model inference using vllm                         
python  email_preprocessing_vllm_unordered.py /path/to/your/data/directory                          
# Create vector database with dedublicated email list 
python create_embeddings.py
```
#### Note: 
/path/to/your/data/directory should be replaced with the actual path where your `.msg` files are stored.
