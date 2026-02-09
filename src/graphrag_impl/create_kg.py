import asyncio, os, sys
from time import time
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship

from concurrent.futures import ThreadPoolExecutor, as_completed
from .llm import llm, embedding_provider
from utils.logging import LOGGER
from dotenv import load_dotenv

load_dotenv()

def create_token_batches(chunks, max_tokens=10000):
    batches = []
    current_batch = []
    current_tokens = 0

    for chunk in chunks:
        chunk_tokens = len(chunk.page_content.split())
        if current_tokens + chunk_tokens > max_tokens and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(chunk)
        current_tokens += chunk_tokens

    if current_batch:
        batches.append(current_batch)
    return batches


def process_chunk(chunk, embedding, doc_transformer, graph, reference_number):
    filename = os.path.splitext(os.path.basename(chunk.metadata["source"]))[0]
    print(chunk.metadata)
    if "seq_num" in chunk.metadata:
        chunk_id = f"{filename}.{chunk.metadata["seq_num"]}"
    else:
        chunk_id = f"{filename}.{hash(chunk.page_content)}"

    # Add the Document and Chunk nodes to the graph
    properties = {
        "filename": filename,
        "Project ID": reference_number,
        "chunk_id": chunk_id,
        "text": chunk.page_content,
        "embedding": embedding
    }

    graph.query("""
        MERGE (d:Document {id: $filename})
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text
        MERGE (d)<-[:PART_OF]-(c)
        WITH c
        CALL db.create.setNodeVectorProperty(c, 'textEmbedding', $embedding)
        """, 
        properties
    )

    # Generate the entities and relationships from the chunk
    graph_docs = doc_transformer.convert_to_graph_documents([chunk])

    # Map the entities in the graph documents to the chunk node
    for graph_doc in graph_docs:
        chunk_node = Node(
            id=chunk_id,
            type="Chunk"
        )

        for node in graph_doc.nodes:

            graph_doc.relationships.append(
                Relationship(
                    source=chunk_node,
                    target=node, 
                    type="DIRECTED"
                    )
                )  
    # add the graph documents to the graph
    graph.add_graph_documents(graph_docs)

async def main():
    if len(sys.argv) != 2:
        LOGGER.error("Usage: python create_kg.py <dir_path>")
        sys.exit(1)

    dir_path = sys.argv[1]

    if not os.path.isdir(dir_path):
        LOGGER.error(f"{dir_path} is not a valid dir.")
        sys.exit(1)

    
    DOCS_PATH = dir_path

    graph = Neo4jGraph(
        url=os.getenv('NEO4J_URI'),
        username=os.getenv('NEO4J_USERNAME'),
        password=os.getenv('NEO4J_PASSWORD')
    )


    prompt =  """
---Entity Descriptions---
- **Code**: Extract any codes mentioned in the email, such as booking numbers, container numbers, truck plates etc.
- **Person**: Extract names of people mentioned in the email, such as contacts, drivers, or any individuals involved.
- **Email_Address**: Extract any email addresses mentioned in the email.
- **Organization**: Extract names of organizations mentioned in the email, such as companies.
- **Operation Phase**: The state of the transportation project. Identify this based on status keywords or the email subject line.    
- **Shipper:** The company that sends or exports the cargo. 
- **Carrier:** The company responsible for transporting the cargo. 
- **Consignee:** The company that receives or imports the cargo.
- **Role:** The role of a company in the transportation project, such as "freight forwarder", "customs broker", "port authority", etc. It can also be used to indicate a person's role, such as "driver", "contact person", "sales executive"tc.
- **Location:** Extract any locations mentioned in the email, such as ports, cities, or countries.
- **Address:** Extract any addresses mentioned in the email, such as pickup or delivery addresses.
- **Date:** Extract any dates mentioned in the email, such as pickup, delivery dates, or date an email was sent.
- **Event:** Extract any events mentioned in the email, such as "pickup", "delivery", "customs clearance", "delay", etc.
- **Transport Name:** Extract the name of the transportation vessel or vehicle, such as the name of a ship or truck.
- **Transportation Service:** Extract the type of transportation service mentioned, such as "sea freight", "sea freight".
- **Cargo Content:** Extract the description of the cargo mentioned in the email, such as "electronics", or measurements.
- **Financial Detail:** Extract any financial details mentioned in the email, such as "cost", "price", "payment terms", etc.
- **Quantity:** Extract any quantities or measurements mentioned in the email, such as "10 containers", "5 pallets", "100 tons", "10x87x90" etc.
- **Attachment:** Extract the names of any attachments mentioned in the email, such as "invoice.pdf", "packing_list.xlsx", etc.
- **Instruction** Extract any instructions mentioned in the email, such as confirmations, requests, custom Formalities, etc.
- **Issue** Extract any issues or problems mentioned in the email, such as "delay", "damage", "missing documents", etc.
- **Concept**: Extract any other relevant concepts mentioned in the email that do not fit into the above categories, such as "incoterms", "transportation regulations", etc.
- Arian Maritime is the freight forwarding company and cannot be of entity type "shipper", "consignee", or "carrier".
"""

    doc_transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=["Code", "Person", "Email_Address", "Organization", "Shipper", "Consignee", "Carrier", "Role", "Location", "Address", "Date", "Event", "Operation_Phase", "Transport_Name", "Transportation_Service", "Cargo_Content", "Financial_Detail", "Quantity", "Attachment", "Instruction", "Issue", "Concept"],
        allowed_relationships=["DIRECTED"],
        node_properties=True,
        relationship_properties=True,
        additional_instructions=prompt
    )

    # Load and split the documents
    json_loader = DirectoryLoader(DOCS_PATH, glob="*unique.json", loader_cls=lambda path: JSONLoader(path, jq_schema=".[]", text_content=False), show_progress=True)

    json_docs = json_loader.load()

    # Produce json embeddings in batches
    json_batches = create_token_batches(json_docs, max_tokens=10000)
    json_tasks = [embedding_provider.aembed_documents([chunk.page_content for chunk in batch]) for batch in json_batches]
    json_embeddings = await asyncio.gather(*json_tasks)
    json_embeddings = [emb for batch in json_embeddings for emb in batch]   # flatten

    # Add Document node to the graph so it will be created once per document
    filename = os.path.splitext(os.path.basename(json_docs[0].metadata["source"]))[0]
    reference_number = os.path.basename(os.path.dirname(json_docs[0].metadata["source"]))


    #for filename in filenames:
    graph.query("""
    MERGE (d:Document {id: $filename}) 
    MERGE (r:ProjectID {id: $referenceNumber})
    MERGE (d)-[:HAS_REFERENCE_NUMBER]->(r)
    """,
    {"filename": filename, "referenceNumber": reference_number}
    )

    # Process the chunks in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(process_chunk, chunk, emb, doc_transformer, graph, reference_number)
            for chunk, emb in zip(json_docs, json_embeddings)
        ]

        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                LOGGER.error(f"Chunk failed: {e}")


    # Create the vector index
    graph.query("""
        CREATE VECTOR INDEX `chunkVector`
        IF NOT EXISTS
        FOR (c: Chunk) ON (c.textEmbedding)
        OPTIONS {indexConfig: {
        `vector.dimensions`: 1024,
        `vector.similarity_function`: 'cosine'
        }};""")
    
    # Add __Entity__ label to all nodes
    graph.query("""
    MATCH (n)
    SET n:__Entity__
    """
    )

    graph.close()
    
if __name__ == "__main__":
    tic1 = time()
    asyncio.run(main())
    LOGGER.info(f"Time taken to process: {time() - tic1} seconds")
