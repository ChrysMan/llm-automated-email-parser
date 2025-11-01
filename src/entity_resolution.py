import os
from typing import List, Optional
from langchain_neo4j import Neo4jVector
from langchain.graphs import Neo4jGraph
from pydantic import BaseModel, Field
from tqdm import tqdm
from graphdatascience import GraphDataScience
from chatbot.llm import llm, qa_llm, cypher_llm, embedding_provider
from chatbot.graph import graph
from langchain_core.prompts import ChatPromptTemplate
from concurrent.futures import ThreadPoolExecutor, as_completed

gds = GraphDataScience(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

# Calculating text embeddings for entities in the knowledge graph
vector = Neo4jVector.from_existing_graph(
    embedding_provider,
    node_label='__Entity__',
    text_node_properties=['id'],
    embedding_node_property='textEmbedding'
)

# Project an in-memory graph to use in gds algorithms
G, result = gds.graph.project(
    "entities",                         # Graph name
    "__Entity__",                       # Node projection
    "*",                                # Relationship projection
    nodeProperties=["textEmbedding"]    # Configuration parameters
)

# Constructing k-nearest graph and storing new relationships in the project graph
similarity_threshold = 0.95

gds.knn.mutate(
    G,
    nodeProperties=['textEmbedding'],
    mutateRelationshipType='SIMILAR',
    mutateProperty='score',
    similarityCutoff=similarity_threshold,
    topK=20
)

# Compute weakly connected components and store the component id as a property on each node
gds.wcc.write(
    G,
    writeProperty="wcc",
    relationshipTypes=["SIMILAR"]
)

# A filter allowing only pairs of words with a text distance of three or fewer
word_edit_distance = 3
potential_duplicate_candidates = graph.query(
    """MATCH (e:`__Entity__`)
    WHERE size(e.id) > 3 // longer than 3 characters
    WITH e.wcc AS community, collect(e) AS nodes, count(*) AS count
    WHERE count > 1
    UNWIND nodes AS node
    // Add text distance
    WITH distinct
      [n IN nodes WHERE apoc.text.distance(toLower(node.id), toLower(n.id)) < $distance 
                  OR node.id CONTAINS n.id | n.id] AS intermediate_results
    WHERE size(intermediate_results) > 1
    WITH collect(intermediate_results) AS results
    // combine groups together if they share elements
    UNWIND range(0, size(results)-1, 1) as index
    WITH results, index, results[index] as result
    WITH apoc.coll.sort(reduce(acc = result, index2 IN range(0, size(results)-1, 1) |
            CASE WHEN index <> index2 AND
                size(apoc.coll.intersection(acc, results[index2])) > 0
                THEN apoc.coll.union(acc, results[index2])
                ELSE acc
            END
    )) as combinedResult
    WITH distinct(combinedResult) as combinedResult
    // extra filtering
    WITH collect(combinedResult) as allCombinedResults
    UNWIND range(0, size(allCombinedResults)-1, 1) as combinedResultIndex
    WITH allCombinedResults[combinedResultIndex] as combinedResult, combinedResultIndex, allCombinedResults
    WHERE NOT any(x IN range(0,size(allCombinedResults)-1,1)
        WHERE x <> combinedResultIndex
        AND apoc.coll.containsAll(allCombinedResults[x], combinedResult)
    )
    RETURN combinedResult
    """, params={'distance': word_edit_distance})

system_prompt = """You are a data processing assistant. Your task is to identify duplicate entities in a list and decide which of them should be merged.
The entities might be slightly different in format or content, but essentially refer to the same thing. Use your analytical skills to determine duplicates.

Here are the rules for identifying duplicates:
1. Entities with minor typographical differences should be considered duplicates.
2. Entities with different formats but the same content should be considered duplicates.
3. Entities that refer to the same real-world object or concept, even if described differently, should be considered duplicates.
4. If it refers to different numbers, dates, or products, do not merge results
"""
user_template = """
Here is the list of entities to process:
{entities}

Please identify duplicates, merge them, and provide the merged list.
"""

class DuplicateEntities(BaseModel):
    entities: List[str] = Field(
        description="Entities that represent the same object or real-world entity and should be merged"
    )


class Disambiguate(BaseModel):
    merge_entities: Optional[List[DuplicateEntities]] = Field(
        description="Lists of entities that represent the same object or real-world entity and should be merged"
    )


llm.with_structured_output(Disambiguate)

extraction_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", user_template),
    ]
).with_partial_variables({"entities": "{entities}"})

extraction_chain = extraction_prompt | llm

def entity_resolution(entities: List[str]) -> Optional[List[List[str]]]:
    return [
        el.entities
        for el in extraction_chain.invoke({"entities": entities}).merge_entities
    ]

merged_entities = []
with ThreadPoolExecutor(max_workers=10) as executor:
    # Submitting all tasks and creating a list of future objects
    futures = [
        executor.submit(entity_resolution, el['combinedResult'])
        for el in potential_duplicate_candidates
    ]

    for future in tqdm(
        as_completed(futures), total=len(futures), desc="Processing documents"
    ):
        to_merge = future.result()
        if to_merge:
            merged_entities.extend(to_merge)

graph.query("""
UNWIND $data AS candidates
CALL {
  WITH candidates
  MATCH (e:__Entity__) WHERE e.id IN candidates
  RETURN collect(e) AS nodes
}
CALL apoc.refactor.mergeNodes(nodes, {properties: {
    description:'combine',
    `.*`: 'discard'
}})
YIELD node
RETURN count(*)
""", params={"data": merged_entities})