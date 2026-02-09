from dotenv import load_dotenv
load_dotenv()
from time import time
from llm import llm
from graph import graph

from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate

CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
-Identify the claims in the question. The Project ID is not considered a claim.
-Translate the claims into a Cypher query that retrieves the relevant information from the graph database.
-Use only the provided relationship types and properties in the schema.
-Do not use any other relationship types or properties that are not provided.
-Do not repeat properties in the RETURN clause. Each property must appear only once.
-Only include the generated Cypher statement in your response.
-Do NOT add any explanation or comments outside the Cypher query.

Always use case insensitive search when matching strings. 

Examples: 
# Use case insensitive matching 
MATCH (c:Chunk)-[:DIRECTED]->(e)
WHERE e.id =~ '(?i).*entityName.*'
RETURN e.id

# Find documents that reference entities
MATCH (d:Document)<-[:PART_OF]-(c:Chunk)-[:DIRECTED]->(e)
WHERE e.id =~ '(?i).*entityName.*'
RETURN d.id, c.id, c.text, e.id

Schema:
{schema}


The question is:
{question}"""


nodes = graph.query("""
    CALL db.schema.nodeTypeProperties() 
    YIELD nodeLabels, propertyName
    RETURN nodeLabels, collect(distinct propertyName) AS properties""")

relationships=graph.query(
    """CALL db.schema.relTypeProperties()
    YIELD relType
    RETURN collect(distinct relType) AS rels""")[0]["rels"]

node_str = "\n".join(
    f"- {','.join(n['nodeLabels'])} (properties: {', '.join(n['properties'])})"
    for n in nodes
)

node_str = node_str.replace("Chunk", "")
rel_str = ", ".join(f"{rel.replace(":","").replace("`","")}" for rel in relationships)

graph_schema = f"""Node Types and their Properties:
{node_str}
Relationship Types:
[DIRECTED, PART_OF, HAS_REFERENCE_NUMBER]"""
#{rel_str}"""

#print("\nGraph Schema:\n", graph_schema)

cypher_generation_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE,
    input_variables=["question"], 
    partial_variables={"schema": graph_schema} 
)

CYPHER_QA_TEMPLATE = """You are an assistant that helps to form nice and human understandable answers.
The information part contains the provided information that you must use to construct an answer.
The provided information is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
Make the answer sound as a response to the question. Do not mention that you based the result on the given information.
Here is an example:

Question: Who does John Doe work for?
Context:['o.id':CTL LLC, 'o.name': None]
Helpful Answer: John Doe works for CTL LLC

Follow this example when generating answers.
Usually the id property holds the information needed.
If the provided information is empty, say that you don't know the answer. 
Information:
{context}

Question: {question}
Answer:"""

cypher_qa_prompt = PromptTemplate(
    input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
)

cypher_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    cypher_prompt=cypher_generation_prompt,
    return_direct=True,
    #qa_prompt=cypher_qa_prompt,
    validate_cypher = True,
    verbose=True,
    allow_dangerous_requests=True,
    top_k=20
)

def run_cypher(q):
    cypher_result =  cypher_chain.invoke({"query": q})
    print("\n\nCypher Result: ", cypher_result)
    return cypher_result.get("result", None) 

# while (q := input("> ")) != "exit":
#     tic1 = time()
#     print(run_cypher(q))
#     print(f"Time taken to process: {time() - tic1} seconds")
  