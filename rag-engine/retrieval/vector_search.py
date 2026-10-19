"""
Vector search via Neo4j Vector Index.
Uses cosine similarity on e5-base-v2 embeddings stored in Neo4j.
"""
from neo4j import GraphDatabase

from models import embed


class VectorSearcher:
    def __init__(self, uri: str, user: str, password: str, index_name: str = "chunk_embeddings"):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._index_name = index_name

    def close(self) -> None:
        self._driver.close()

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Return top_k chunks ranked by cosine similarity to query embedding."""
        query_embedding = embed([f"query: {query}"])[0]

        cypher = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
        YIELD node AS chunk, score
        RETURN
            chunk.chunk_id     AS chunk_id,
            chunk.content      AS content,
            chunk.spec_id            AS spec_id,
            chunk.section_title      AS section,
            score
        ORDER BY score DESC
        """

        with self._driver.session() as session:
            result = session.run(
                cypher,
                index_name=self._index_name,
                top_k=top_k,
                embedding=query_embedding,
            )
            return [dict(record) for record in result]
