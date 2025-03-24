# Automating Information Extraction from Emails using Large Language Models

*This project is still in progress* 

## Prerequisites
- Python 3.12 installed
- `ollama` installed ([Installation Guide](https://ollama.ai/))
- `pip` installed (comes with Python)
- `.msg` email files available for processing




## Setup & Execution:

### Create a Virtual Environment:
Run the following command in your project's root directory:

#### For Linux/macOS: 
```sh
python3 -m venv my_project_env
```
#### For Windows: 
```sh
python -m venv my_project_env
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
#### For Windows (PowerShell): 
```sh
(venv) user@machine:~/project$
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
python app.py /path/to/your/data/directory
```
#### Note: 
/path/to/your/data/directory should be replaced with the actual path where your `.msg` files are stored.