import os
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv

load_dotenv()

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

print("Connected to Neo4j Sandbox!")

graph.query("""
MERGE (a:Person {name: 'Alice'})
MERGE (b:Person {name: 'Bob'})
MERGE (a)-[:KNOWS]->(b)
""")

print("Created nodes and relationship!")

result = graph.query("""
MATCH (a:Person)-[:KNOWS]->(b:Person)
RETURN a.name AS from, b.name AS to
""")

print("✅ Query result:")
for record in result:
    print(f"{record['from']} → {record['to']}")
