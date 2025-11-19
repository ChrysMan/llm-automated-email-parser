import asyncio, os, sys
import pytesseract
import tempfile
import pandas as pd
from utils.graph_utils import write_file
from time import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.logging_config import LOGGER
from langchain_community.document_loaders import DirectoryLoader, TextLoader, JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from PIL import Image
from pytesseract import Output

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

def extract_text_with_ocr(path)->pd.DataFrame:
    """Convert PDF or JPGs, JPEGs, PNGs pages to images and run OCR."""
    all_data = []
    final_df = pd.DataFrame()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            if path.lower().endswith((".jpg", ".jpeg", ".png")):
                img = Image.open(path)
                data = pytesseract.image_to_data(img, lang="eng+ell+chi_sim+rus+hun", output_type=Output.DATAFRAME)
                #df = pd.read_csv(io.StringIO(data), sep='\t')
                all_data.append(data)
                #text += img
            elif path.lower().endswith(".pdf"):
                images = convert_from_path(path, output_folder=temp_dir)
                for i, img in enumerate(images):
                    data = pytesseract.image_to_data(img, lang="eng+ell", output_type=Output.DATAFRAME)
                    #df = pd.read_csv(io.StringIO(data), sep='\t')
                    #df['page'] = i + 1
                    all_data.append(data)
                    #text += f"{page_text.strip()}\n"
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            final_df = final_df[final_df.conf != -1].dropna(subset=['text'])  # filter out invalid data, drop rows with NaN text
        else:
            final_df = pd.DataFrame()
    except Exception as e:
        print(f"OCR failed for {path}: {e}")
    return final_df

def process_all_pdfs(root_path):
    """Recursively find all PDFs or JPGs, JPEGs, PNGs and extract text via OCR."""
    ocr_results = {}
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.lower().endswith(".pdf") or filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg") or filename.lower().endswith(".png"):
                file_path = os.path.join(dirpath, filename)
                print(f"Processing {file_path} ...")
                ocr_results[file_path] = extract_text_with_ocr(file_path)
    return ocr_results

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
        "referenceNumber": reference_number,
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
    # model = "qwen2.5:14b"
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
        ("Chunk", "HAS_ENTITY", "EmailAddress"),
        ("Chunk", "HAS_ENTITY", "Person"),
        ("Chunk", "HAS_ENTITY", "Reference_Number"),
        ("Chunk", "SENT_BY", "EmailAddress"),
        ("Chunk", "SENT_TO", "EmailAddress"),
        ("Chunk", "CC_TO", "EmailAddress"),
        ("Chunk", "SENT_AT", "Date"),
        ("Chunk", "SENT_BY", "Person"),
        ("Chunk", "SENT_TO", "Person"),
        ("Chunk", "CC_TO", "Person"),
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
You are extracting a knowledge graph from email text in json format or from PDFs invoices scanned into txt format. 
Prefer using the following schema when identifying entities and relationships:

Entities: Person, Organization, Location, Date, Destination, Airport, Port, Product, Route, Attachment, Shipper, Shipment
Relationships: WORKS_AT, SENT_TO, CCED, SHIPPED_BY, RECEIVED_BY, CORRESPONDENCE, HAS_ENTITY

Rules:
- You may define a new entity or relationship type ONLY if it clearly adds new information not represented by the above list.
- When introducing a new type, use descriptive names and keep them consistent across messages. 
- Extract as many usefull information as possible from the email body text to populate the knowledge graph.
- For additional information related to an entity (e.g. email address of a person, product ID, quantity, departure/arrival date of a shipment), store it as a property of the corresponding node instead of creating a separate node.
- Every node needs to have a unique identifier property 'id'.

Extra rules for extracting from scanned PDF invoices in txt format:
- The txt where extracted from scanned PDFs may contain OCR errors. Use context to correctly identify entities and relationships despite minor text inaccuracies.
- The text was created by grouping the OCR output by block_num, par_num, and line_num.

I provide you with some acronyms and their meanings that are commonly used in shipping emails.Use them to understand the context better and extract more accurate information:
ACY: AGENCY 
AGN: AGENT	
SHP: SHIPPER	
CNE: CONSIGNEE	
CL: CLIENT	
TR: TRUCKER	
INC: INCURANCE	
BL: BILL OF LADING
BLF, HBLF, MBLF: BILL OF LADING FINAL
BLD, HBLD, MBLD: BILL OF LADING DRAFT
BLO, HBLO, MBLO: BILL OF LADING ORIGINAL
DN: DEBIT NOTE
CN: CREDIT NOTE
COMD: COMMERCIAL DOCS
PAD: PRE ADVISE DOCS
PADD: PRE ADVISE DOCS * DRAFTS
SI: SHIPPING INSTRUCTIONS / NOTE
BN: BOOKING NOTE
TICS: TICKETS 
VCHR: VOUCHER 
AN: ARRIVAL NOTICE
DO: DELIVERY ORDER
BA: BOOKING ASSIGNMENT
SR: SELLING RATE
BR: BUYING RATE
RO: RELEASE ORDER
COR: CORRESPONDENCE
PAL: PRE ADVISE
BC: BOOKING CONFIRMATION

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
  properties: id = "John Doe", email = "john.doe@globalfreight.com", organization = "Global Freight Ltd."
Person: Maria Rossi
  properties: id = "Maria Rossi", email = "m.rossi@seashipments.it", organization = "SeaShipments Italia S.p.A."
Organization: Global Freight Ltd.
  properties: id = "Global Freight Ltd", location = "Port of Piraeus, Greece"
Organization: SeaShipments Italia S.p.A.
  properties: id = "SeaShipments Italia S.p.A.", location = "Port of Shanghai"
Shipment: 123456
  properties: 
    id = "123456"
    products = ["OceanAir pressure valves"]
    quantity = [250]
    product_ids = ["OA-532"]
    departure_date = "June 5th"
    arrival_date = "June 20th"
    shipper_organizations = ["Global Freight Ltd.", "SeaShipments Italia S.p.A."]
    from_port = "Port of Piraeus"
    to_port = "Port of Shanghai"
Date: Tuesday, May 30, 2023 10:36 AM
    properties: id = "Tuesday, May 30, 2023 10:36 AM"
Organization: Global Freight Ltd.
    properties: id = "Global Freight Ltd."
Organization: SeaShipments Italia S.p.A.
    properties: id = "SeaShipments Italia S.p.A."
Attachment: MBL
    properties: id = "MBL"
Attachment: HBL
    properties: id = "HBL"

Edges:
  - (John Doe) SENT_TO (Maria Rossi)
  - (John Doe) CCED (logistics@globalfreight.com)
  - (John Doe) CORRESPONDENCE (Maria Rossi)
  - (Maria Rossi) WORKS_AT (SeaShipments Italia S.p.A.)
  - (John Doe) WORKS_AT (Global Freight Ltd.)
  - (Global Freight Ltd.) SHIPPER_OF (Shipment_Reference_Number: 123456)
  - (123456) SHIPPED_BY (Global Freight Ltd.)
  - (123456) RECEIVED_BY (SeaShipments Italia S.p.A.)
  - (123456) SHIPS_TO (Port of Shanghai)
  - (123456) SHIPPED_FROM (Port of Piraeus)
"""


    doc_transformer = LLMGraphTransformer(
        llm=llm,
        #allowed_nodes=["Document", "Person", "Organization", "Company", "Entity", "Team", "Location", "Date", "EmailAddress", "EmailChunk", "Reference_Number", "Destination", "Shipment", "Airport", "Port", "Product", "Shipper", "Route"],
        #allowed_relationships=allowed_relationships2,
        node_properties=True,
        relationship_properties=True,
        additional_instructions=prompt
    )

    # Process all PDFs and JPGs, JPEGs, PNGs  with OCR and save the text files
    result = process_all_pdfs(DOCS_PATH)
    for file_path, df in result.items():
        if isinstance(df, pd.DataFrame):
            lines = []
            for _, group in df.groupby(["block_num", "par_num", "line_num"]):
                line_text = " ".join(group["text"].dropna().tolist())
                lines.append(line_text)
            text_blob = "\n".join(lines)
            out_file = file_path.replace(".pdf", "_ocr.txt").replace(".jpg", "_ocr.txt").replace(".jpeg", "_ocr.txt").replace(".png", "_ocr.txt")
            write_file(text_blob, out_file)

    # Load and split the documents
    json_loader = DirectoryLoader(DOCS_PATH, glob="*unique.json", loader_cls=lambda path: JSONLoader(path, jq_schema=".[]", text_content=False), show_progress=True)
    txt_loader = DirectoryLoader(DOCS_PATH, glob="*.txt", loader_cls=TextLoader, use_multithreading=True)

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
    )

    json_docs = json_loader.load()
    txt_docs = txt_loader.load()
    txt_chunks = text_splitter.split_documents(txt_docs)

    # Produce json embeddings in batches
    json_batches = create_token_batches(json_docs, max_tokens=10000)
    json_tasks = [embedding_provider.aembed_documents([chunk.page_content for chunk in batch]) for batch in json_batches]
    json_embeddings = await asyncio.gather(*json_tasks)
    json_embeddings = [emb for batch in json_embeddings for emb in batch]   # flatten

    # Produce txt embeddings in batches
    txt_batches = create_token_batches(txt_chunks, max_tokens=10000)
    txt_tasks = [embedding_provider.aembed_documents([chunk.page_content for chunk in batch]) for batch in txt_batches]
    txt_embeddings = await asyncio.gather(*txt_tasks)
    txt_embeddings = [emb for batch in txt_embeddings for emb in batch]     # flatten

    # Combine the chunks and embeddings
    final_chunks = json_docs + txt_chunks
    final_embeddings = json_embeddings + txt_embeddings
    #print([final_chunk for final_chunk in final_chunks[17:]])

    # Add Document node to the graph so it will be created once per document
    filename = os.path.splitext(os.path.basename(json_docs[0].metadata["source"]))[0]
    reference_number = os.path.basename(os.path.dirname(json_docs[0].metadata["source"]))

    filenames = list(set([os.path.splitext(os.path.basename(txt_chunk.metadata["source"]))[0] for txt_chunk in txt_chunks]))
    filenames.append(filename)

    for filename in filenames:
        graph.query("""
        MERGE (d:Document {id: $filename}) 
        MERGE (r:ReferenceNumber {id: $referenceNumber})
        MERGE (d)-[:HAS_REFERENCE_NUMBER]->(r)
        """,
        {"filename": filename, "referenceNumber": reference_number}
        )

    # Process the chunks in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(process_chunk, chunk, emb, doc_transformer, graph, reference_number)
            for chunk, emb in zip(final_chunks, final_embeddings)
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
        `vector.dimensions`: 768,
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
