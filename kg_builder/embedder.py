"""
Embedder — computes e5-base-v2 embeddings for all Chunk nodes
and creates a Neo4j vector index for cosine similarity search.
"""
import os

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "intfloat/e5-base-v2"
VECTOR_INDEX_NAME = "chunk_embeddings"
EMBEDDING_DIM = 768
# Batch size for encode + bulk-write. Large to exploit GPU (e5-base is light, 16GB
# VRAM is plenty). Override via env EMBED_BATCH_SIZE to lower on weak machines.
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "256"))


def _passage_text(section_title: str | None, content: str | None) -> str:
    """Build the e5 passage string from section_title + content.

    section_title goes before content because it condenses the chunk topic
    ("Definition of AMF", "N6 Reference Point") — feeding this signal into vector
    space improves recall. Skip the title when empty or already a prefix of
    content (avoid duplicating the phrase twice).
    """
    title = (section_title or "").strip()
    body = (content or "").strip()
    if not title or body.startswith(title):
        return f"passage: {body}"
    return f"passage: {title}\n{body}"


class Embedder:
    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        self._uri = uri or os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        self._user = user or os.getenv("NEO4J_USER", "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD", "password")
        self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
        self._model: SentenceTransformer | None = None

    def close(self) -> None:
        self._driver.close()

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            # Pick device: cuda if GPU present, else cpu. Override via EMBED_DEVICE.
            import torch
            device = os.getenv("EMBED_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[embed] Loading {EMBEDDING_MODEL} on {device}...")
            self._model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        return self._model

    def create_vector_index(self) -> None:
        """Create Neo4j vector index if not exists."""
        with self._driver.session() as s:
            s.run(
                f"""
                CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {{indexConfig: {{
                  `vector.dimensions`: {EMBEDDING_DIM},
                  `vector.similarity_function`: 'cosine'
                }}}}
                """
            )
        print(f"[embed] Vector index '{VECTOR_INDEX_NAME}' ready.")

    def clear_embeddings(self) -> int:
        """Remove the `embedding` property from every Chunk. Must be called before
        re-embedding everything when the passage formula changes (e.g. adding
        section_title) — otherwise embed_all_chunks() only touches NULL chunks and
        the vector space mixes old and new formulas. Returns count of embeddings cleared."""
        with self._driver.session() as s:
            rec = s.run(
                "MATCH (c:Chunk) WHERE c.embedding IS NOT NULL "
                "REMOVE c.embedding RETURN count(c) AS n"
            ).single()
        n = rec["n"] if rec else 0
        print(f"[embed] Cleared embeddings on {n} chunks (will re-embed).")
        return n

    def embed_all_chunks(self) -> int:
        """Embed every Chunk node that has no embedding yet. Returns count embedded."""
        model = self._get_model()

        # Also fetch section_title — it is a very strong retrieval signal
        # ("N6 Reference Point", "Definition of AMF") but was previously outside
        # the vector space (only prepended at rerank time). Feed it into the
        # passage so it participates in vector search over the ENTIRE dataset.
        with self._driver.session() as s:
            rows = s.run(
                "MATCH (c:Chunk) WHERE c.embedding IS NULL "
                "RETURN c.chunk_id AS id, c.section_title AS section_title, c.content AS content"
            ).data()

        if not rows:
            print("[embed] All chunks already have embeddings.")
            return 0

        print(f"[embed] Embedding {len(rows)} chunks in batches of {BATCH_SIZE}...")
        total = 0

        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i : i + BATCH_SIZE]
            # Prepend section_title into the passage. The query at inference keeps
            # `query: {text}` as-is (see vector_search) — only the passage side changes.
            texts = [_passage_text(r["section_title"], r["content"]) for r in batch]
            embeddings = model.encode(
                texts, normalize_embeddings=True, batch_size=BATCH_SIZE
            ).tolist()

            # Bulk-write the whole batch in one UNWIND query instead of one
            # s.run()/chunk — cuts Neo4j round-trips from N to 1 per batch.
            params = [
                {"id": row["id"], "emb": emb}
                for row, emb in zip(batch, embeddings)
            ]
            with self._driver.session() as s:
                s.run(
                    """
                    UNWIND $rows AS row
                    MATCH (c:Chunk {chunk_id: row.id})
                    SET c.embedding = row.emb
                    """,
                    rows=params,
                )

            total += len(batch)
            pct = total / len(rows) * 100
            print(f"[embed] {total}/{len(rows)} ({pct:.0f}%)", end="\r")

        print(f"\n[embed] Done — {total} chunks embedded.")
        return total
