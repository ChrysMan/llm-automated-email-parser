from llm import llm
from graph import graph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.tools import Tool
from langchain_neo4j import Neo4jChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
from utils import get_session_id

from tools.vector import find_chunk
from tools.cypher import run_cypher

# chat_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are an AI expert providing information about Neo4j and Knowledge Graphs."),
#         ("human", "{input}"),
#     ]
# )

# kg_chat = chat_prompt | llm | StrOutputParser()

# tag::tools[]
tools = [
    # Tool.from_function(
    #     name="General Assistant",
    #     description="""Use this for conversations that do NOT require accessing the email content 
    #         or the knowledge graph. This includes general questions, clarifications, 
    #         or reasoning without retrieving data.""",
    #     func=kg_chat.invoke,
    # ), 
    Tool.from_function(
        name="Email Content Search",
        description="Use this when the answer requires reading the content of emails, such as sender, date, or message text.",
        func=find_chunk, 
    ),
    Tool.from_function(
        name="Knowledge Graph information",
        description="Use this when the answer requires structured entity/relationship info from the knowledge graph.",
        func = run_cypher,
    )
]
# end::tools[]

def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

# tag::agent_prompt[]
agent_prompt = PromptTemplate.from_template("""
You are an AI Assistant specialized in answering questions about the company's email data using the Knowledge Graph and available tools.
Be as helpful as possible and return as much information as possible.
        
Always use a tool and only use the information provided in the context.
If multiple tools are relevant, you can use them sequentially. Always choose the most appropriate tool for each part of the question.

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
""")
# end::agent_prompt[]

agent = create_react_agent(llm, tools, agent_prompt)
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

    response = chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": get_session_id()}},)

    return response['output']
