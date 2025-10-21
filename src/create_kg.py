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
    filename = os.path.basename(email_chunk.metadata["source"])
    #chunk_id = f"{filename}.{chunk.metadata["seq_num"]}"
    chunk_id = f"{filename}.{email_chunk.metadata.get('page', 0)}"
    #LOGGER.info("Processing -", chunk_id)
    
    # Add the Document and Chunk nodes to the graph
    properties = {
        "filename": filename,
        "chunk_id": chunk_id,
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
                )    # Embed the chunk

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

    doc_transformer = LLMGraphTransformer(
        llm=llm,
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
    json_chunks = text_splitter.split_documents(json_docs)

    # Produce embeddings in batches
    batches = create_token_batches(json_docs, max_tokens=10000)
    tasks = [embedding_provider.aembed_documents([chunk.page_content for chunk in batch]) for batch in batches]
    embeddings = await asyncio.gather(*tasks)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(process_chunk, chunk, emb, doc_transformer, graph)
            for chunk, emb in zip(json_chunks, embeddings)
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
