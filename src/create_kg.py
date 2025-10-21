import asyncio, os, sys
from time import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.logging_config import LOGGER
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship

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

def process_chunk(email_chunk, embedding, doc_transformer, graph):
    filename = os.path.splitext(os.path.basename(email_chunk.metadata["source"]))[0]
    chunk_id = f"{filename}.{email_chunk.metadata["seq_num"]}"
    #doc_id = os.path.splitext(filename)[0]
    #email_chunk.metadata["doc_id"] = doc_id

    # Add the Document and Chunk nodes to the graph
    properties = {
        "filename": filename,
        "chunk_id": chunk_id,
        #"doc_id" : doc_id,
        "text": email_chunk.page_content,
        "embedding": embedding
    }
    
    graph.query("""
        MERGE (d:Document {id: $filename})
        MERGE (c:EmailChunk {id: $chunk_id})
        SET c.text = $text
        MERGE (d)<-[:PART_OF]-(c)
        WITH c
        CALL db.create.setNodeVectorProperty(c, 'textEmbedding', $embedding)
        """, 
        properties
    )

    # Generate the entities and relationships from the chunk
    graph_docs = doc_transformer.convert_to_graph_documents([email_chunk])

    # Map the entities in the graph documents to the chunk node
    for graph_doc in graph_docs:
        chunk_node = Node(
            id=chunk_id,
            type="EmailChunk"
        )

        for node in graph_doc.nodes:

            graph_doc.relationships.append(
                Relationship(
                    source=chunk_node,
                    target=node, 
                    type="HAS_ENTITY"
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

    # model = "mistral-7b"
    # model = "neural-chat:latest"
    # model = "gemma3:latest"
    # model = "llama3.2:3b"
    # model = "llama3.1:8b"
    model = "qwen2.5:14b"

    llm = ChatOllama( 
        model=model,
        temperature=0
    )

    #embed_model = "mxbai-embed-large"
    embed_model = "nomic-embed-text"

    embedding_provider = OllamaEmbeddings(
        model=embed_model
    )

    graph = Neo4jGraph(
        url=os.getenv('NEO4J_URI'),
        username=os.getenv('NEO4J_USERNAME'),
        password=os.getenv('NEO4J_PASSWORD')
    )

    allowed_relationships1 = [
        ("EmailChunk", "HAS_ENTITY", "EmailAddress"),
        ("EmailChunk", "HAS_ENTITY", "Person"),
        ("EmailChunk", "HAS_ENTITY", "Reference_Number"),
        ("EmailChunk", "SENT_BY", "EmailAddress"),
        ("EmailChunk", "SENT_TO", "EmailAddress"),
        ("EmailChunk", "CC_TO", "EmailAddress"),
        ("EmailChunk", "SENT_AT", "Date"),
        ("EmailChunk", "SENT_BY", "Person"),
        ("EmailChunk", "SENT_TO", "Person"),
        ("EmailChunk", "CC_TO", "Person"),
        ("Person", "OWNER_OF", "EmailAddress"),
        ("Person", "WORKS_AT", "Organization"),
        ("Person", "CORRESPONDENCE", "Person"),
        ("Organization", "SHIPS_TO", "Airport"),
        ("Organization", "SHIPS_TO", "Port"),
        ("Shipment", "SHIPPED_AT", "Date")
    ]

    allowed_relationships2 = [
        "HAS_ENTITY",
        "WORKS_AT",
        "LOCATED_IN",
        "RELATED_TO",
        "PART_OF",
        "CONTAINS",
        "MENTIONS",
        "SENT_TO",
        "SENT_BY",
        "SENT_AT",
        "CC_TO",
        "OWNER_OF",
        "SHIPPER_OF",
        "SHIPS_TO",
        "SHIPPED_AT",
        "RECEIVER_OF",
        "CORRESPONDENCE" 
    ]

    prompt =  """
You are extracting a knowledge graph from email text. 
Prefer using the following schema when identifying entities and relationships:

Entities: Document, Person, Organization, Location, Date, EmailAddress, Destination, Airport, Port, Product, Route, Attachment
Relationships: HAS_ENTITY, WORKS_AT, SENT_TO, CCED, SHIPPED_BY, RECEIVED_BY, CORRESPONDENCE

Rules:
- Document node and EmailChunk nodes have been already created in the graph database, no need to create them again.
- You may define a new entity or relationship ONLY if it clearly adds new information not represented by the above list.
- When introducing a new type, use descriptive names and keep them consistent across messages. 
- Extract as many usefull information as possible from the email body text to populate the knowledge graph.
- For additional information related to an entity (e.g., email address of a person, product ID, quantity, departure/arrival date of a shipment), store it as a property of the corresponding node instead of creating a separate node.

Example Input:
{{  
    "from": "john.doe@globalfreight.com"
    "sent": "Tuesday, May 30, 2023 10:36 AM"
    "to": "m.rossi@seashipments.it"
    "cc": "logistics@globalfreight.com"
    "subject": "Shipment ref: 123456 – Greece-China"
    "body": "Dear Maria Rossi,
    please find attached draft MBL and HBL for shipment ref: 123456. The shipment containing 250 units of the OceanAir pressure valves (Product ID: OA-532) are scheduled to depart from Piraeus on June 5th and arrive in Shanghai on June 20th. 
    Our shipper, Global Freight Ltd., coordinated the delivery route in partnership with SeaShipments Italia S.p.A. Please notify us if there are any customs issues upon arrival.
    Best regards,
    John Doe
    "
}}

Example Output (simplified pseudo-graph):
Nodes:
Person: John Doe
  properties: email = "john.doe@globalfreight.com", organization = "Global Freight Ltd."
Person: Maria Rossi
  properties: email = "m.rossi@seashipments.it", organization = "SeaShipments Italia S.p.A."
Organization: Global Freight Ltd.
  properties: location = "Port of Piraeus, Greece"
Organization: SeaShipments Italia S.p.A.
  properties: location = "Port of Shanghai"
Shipment: 123456
  properties: 
    products = ["OceanAir pressure valves"]
    quantity = [250]
    product_ids = ["OA-532"]
    departure_date = "June 5th"
    arrival_date = "June 20th"
    shipper_organizations = ["Global Freight Ltd.", "SeaShipments Italia S.p.A."]
    from_port = "Port of Piraeus"
    to_port = "Port of Shanghai"
Date: Tuesday, May 30, 2023 10:36 AM
Organization: Global Freight Ltd.
Organization: SeaShipments Italia S.p.A.
EmailAddress: john.doe@globalfreight.com
EmailAddress: m.rossi@seashipments.it
Attachment: MBL
Attachment: HBL

Edges:
  - (John Doe) SENT_TO (Maria Rossi)
  - (John Doe) CCED (logistics@globalfreight.com)
  - (John Doe) CORRESPONDENCE (Maria Rossi)
  - (Maria Rossi) WORKS_AT (SeaShipments Italia S.p.A.)
  - (John Doe) WORKS_AT (Global Freight Ltd.)
  - (Global Freight Ltd.) SHIPPER_OF (Shipment_Reference_Number: 123456)
  - (123456) SHIPPED_BY (Global Freight Ltd.)
  - (123456) RECEIVED_BY (SeaShipments Italia S.p.A.)
   """

    # - Each document has a unique doc_id.
    # - Create only one node of type "Document" per doc_id.
    # - All entities and relationships from chunks belonging to the same doc_id must connect to this single Document node.
    # - Never create multiple Document nodes for the same doc_id.

    doc_transformer = LLMGraphTransformer(
        llm=llm,
        #allowed_nodes=["Document", "Person", "Organization", "Company", "Entity", "Team", "Location", "Date", "EmailAddress", "EmailChunk", "Reference_Number", "Destination", "Shipment", "Airport", "Port", "Product", "Shipper", "Route"],
        #allowed_relationships=allowed_relationships2,
        node_properties=True,
        relationship_properties=True,
        additional_instructions=prompt
    )

    # Load and split the documents
    json_loader = DirectoryLoader(DOCS_PATH, glob="*unique.json", loader_cls=lambda path: JSONLoader(path, jq_schema=".[]", text_content=False), show_progress=True)
    #pdf_loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True, use_multithreading=True)

    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1500,
        chunk_overlap=200,
    )

    json_docs = json_loader.load()
    #json_chunks = text_splitter.split_documents(json_docs)

    # Produce embeddings in batches
    batches = create_token_batches(json_docs, max_tokens=10000)
    tasks = [embedding_provider.aembed_documents([chunk.page_content for chunk in batch]) for batch in batches]
    embeddings = await asyncio.gather(*tasks)
    embeddings = [emb for batch in embeddings for emb in batch]

    # Add Document node to the graph so it will be created once per document
    doc_id = os.path.splitext(os.path.basename(json_docs[0].metadata["source"]))[0]
    reference_number = doc_id.split("-")[0]  
    graph.query("""MERGE (d:Document {id: $doc_id}) SET d.referenceNumber = $referenceNumber""", {"doc_id": doc_id, "referenceNumber": reference_number})

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(process_chunk, chunk, emb, doc_transformer, graph)
            for chunk, emb in zip(json_docs, embeddings)
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
        FOR (c: EmailChunk) ON (c.textEmbedding)
        OPTIONS {indexConfig: {
        `vector.dimensions`: 768,
        `vector.similarity_function`: 'cosine'
        }};""")

    
if __name__ == "__main__":
    tic1 = time()
    asyncio.run(main())
    LOGGER.info(f"Time taken to process: {time() - tic1} seconds")
