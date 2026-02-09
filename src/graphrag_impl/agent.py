from llm import llm
from graph import graph
from langchain_core.prompts import PromptTemplate

from langchain.tools import tool
from langchain_neo4j import Neo4jChatMessageHistory
from langchain_classic.agents import AgentExecutor
from langchain.agents import create_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from utils import get_session_id

from vector import find_chunk
from cypher import run_cypher

@tool("Email Content Search")
def email_content_search(user_input: str) -> str:
    """
    Retrieve specific information within the emails, such as keywords, 
    sender names, date ranges, or any other relevant information that may be contained in the email content.

    Args:
        user_input (str): The user input

    Returns:
        str: The most relevant email content that matches the query.
    """
    return find_chunk(user_input)

@tool("Knowledge Graph information")
def knowledge_graph_information(user_input: str) -> str:
    """
    Retrieve structured information about entities, relationships, and other metadata stored in the graph.
    
    Args:
        user_input (str): The user input.

    Returns:
        str: Relevant entities and relationships from the knowledge graph based on the user input.
    """
    return run_cypher(user_input)

tools = [email_content_search, knowledge_graph_information]

def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

# tag::agent_prompt[]
#PromptTemplate.from_template(
agent_prompt = """
You are an AI Assistant specialized in answering questions about the company's email data using the Knowledge Graph and available tools.
Be as helpful as possible and return as much information as possible.
        
Always use a tool and only use the information provided in the context.
If multiple tools are relevant, you can use them sequentially. Always choose the most appropriate tool for each part of the question.

TOOLS:
------

You have access to the following tools:

1. Email Content Search: Use this tool to search the content of emails based on a query. The query can include keywords, sender names, date ranges, or any other relevant information. The tool will return the most relevant email content that matches the query.
2. Knowledge Graph information: Use this tool to retrieve structured information about entities, relationships, and other metadata stored in the graph. The input of the tool should be the user input.

To use a tool, please use the following reasoning format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [Email Content Search, Knowledge Graph information]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```
"""

prompt = """You are an Enterprise Supervisor Agent managing a knowledge graph and RAG pipeline for maritime corporation email threads. You use tools to answer the employees queries.

GOAL:
Use the outputs from the tools to generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Knowledge Graph.

TOOLS:
1. Email Content Search: Use this tool to search the content of emails based on a query. The query can include keywords, sender names, date ranges, or any other relevant information. The tool will return the most relevant email content that matches the query.
2. Knowledge Graph information: Use this tool to query the knowledge graph using Cypher. You can use this to retrieve structured information about entities, relationships, and other metadata stored in the graph. The input of the tool should be the user input, and the output will be the results of that query.

RULES:
1. Use both tools to answer the question. If the question can be answered using only one tool, use that tool. If the question requires information from both tools, use them sequentially.
3. Strictly adhere to the output from the tools; DO NOT invent, assume, or infer any information not explicitly stated.
4. If the answer cannot be found even after you used both tools, state that you do not have enough information to answer. Do not attempt to guess.

FORMATTING & LANGUAGE:
1. The response MUST be in the same language as the user query.
2. The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).

OUTPUT:
1. Answer only what is explicitly asked. Do not include additional information that is not directly relevant to the question.
2. If you don't have enough information to answer the question, say "I don't have enough information to answer that question."
"""
# Previous conversation history:
# {chat_history}

# New input: {input}
# {agent_scratchpad}

# Begin!

agent = create_agent(llm, tools=tools, system_prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
    verbose=True
    )

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = agent.invoke(
         {"messages": [{"role": "user", "content": user_input}]}
        # {"input": user_input},
        # {"configurable": {"session_id": get_session_id()}},)
    )
    print("\n\nresponse: ", response)
    return response['messages'][-1].content
