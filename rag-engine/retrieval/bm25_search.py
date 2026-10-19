"""
BM25 (sparse) search via Neo4j full-text index.

Neo4j's full-text indexes are Lucene-backed and score with BM25 by default, so
`db.index.fulltext.queryNodes` gives a genuine BM25 baseline over the same Chunk
nodes the dense/vector branch uses — no extra library, no separate index store.
Mirrors VectorSearcher's shape so the orchestrator can treat the two branches
symmetrically (RRF fusion in the BM25+Dense hybrid mode).

The index (`chunk_fulltext` on Chunk.content + section_title) is declared in
kg_builder/builder.py:CYPHER_INDEXES; ensure_index() also self-heals it at
startup so a KG built before this feature still works.
"""
import os
import re

from neo4j import GraphDatabase

# Lucene's query parser treats +-&&||!(){}[]^"~*?:\/ as operators. 3GPP questions
# are full of them ("TS 23.502?", "N6/N9", "Nudm_SDM_Get:"). Rather than escape,
# reduce the question to a bag of tokens — that is exactly the BM25 query model
# (term frequency over a term set) and cannot trip the parser.
#
# The underscore MUST survive that reduction. Index-side, Neo4j's default
# `standard` analyzer follows UAX#29 where "_" is ExtendNumLet, i.e. a JOINER —
# `Nnwdaf_MLModelMonitor_Register` is ONE indexed token. The original pattern
# below split it query-side into three, so no SBI operation name could ever
# match and the whole sparse baseline was crippled on entity-lookup questions.
# Underscore is not a Lucene operator, so keeping it needs no escaping.
#
# Measured on wh_kg_v5_600 (150-question sample, raw index, no reranker):
#   legacy               R@5 0.622  MRR 0.608
#   whole (this pattern) R@5 0.799  MRR 0.786   +0.178 MRR, 34 wins / 0 losses
# "." and "-" were measured too and are NOT joined: keeping them costs
# -0.012 MRR (6 losses / 2 wins), so `TS 23.502` still splits to `ts 23 502`.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*")
_LEGACY_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# `legacy` reproduces the pre-fix behaviour exactly — it is what lets an older
# run be re-measured and what supports a "how much was the bug worth" ablation.
QUERY_MODE = os.getenv("BM25_QUERY_MODE", "whole")

INDEX_NAME = "chunk_fulltext"

# Single source of truth for the DDL. kg_builder.CYPHER_INDEXES carries a copy
# (it is the canonical build-time declaration); tests/test_bm25_query.py asserts
# the two never drift, because a mismatch would silently give a fresh KG one
# analyzer and an existing KG another — CREATE ... IF NOT EXISTS never alters.
FULLTEXT_DDL = (
    f"CREATE FULLTEXT INDEX {INDEX_NAME} IF NOT EXISTS "
    "FOR (c:Chunk) ON EACH [c.content, c.section_title]"
)


def _lucene_query(text: str, mode: str | None = None) -> str:
    """Question -> space-joined Lucene terms (bare OR-of-terms, no operators)."""
    rx = _LEGACY_TOKEN_RE if (mode or QUERY_MODE) == "legacy" else _TOKEN_RE
    return " ".join(rx.findall(text.lower()))


class BM25Searcher:
    def __init__(self, uri: str, user: str, password: str, index_name: str = INDEX_NAME):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._index_name = index_name

    def close(self) -> None:
        self._driver.close()

    def ensure_index(self) -> None:
        """Create the full-text index if a rebuild predating this feature dropped it."""
        cypher = FULLTEXT_DDL.replace(INDEX_NAME, self._index_name)
        with self._driver.session() as session:
            session.run(cypher)

    def _query(self, lucene: str, top_k: int) -> list[dict]:
        # NB: parameter is $search, not $query — session.run()'s first positional
        # arg is itself named `query`, so a `query=` kwarg collides with it.
        cypher = """
        CALL db.index.fulltext.queryNodes($index_name, $search, {limit: $top_k})
        YIELD node AS chunk, score
        RETURN
            chunk.chunk_id     AS chunk_id,
            chunk.content      AS content,
            chunk.spec_id      AS spec_id,
            chunk.section_title AS section,
            score
        ORDER BY score DESC
        """
        with self._driver.session() as session:
            result = session.run(
                cypher,
                index_name=self._index_name,
                search=lucene,
                top_k=top_k,
            )
            return [dict(record) for record in result]

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Return top_k chunks ranked by BM25 relevance to the query text."""
        lucene = _lucene_query(query)
        if not lucene:
            return []
        rows = self._query(lucene, top_k)
        # Underscore-joined terms are rarer than their parts, so in principle a
        # question could now match nothing where the split form matched something.
        # Measured 0/150 on wh_kg_v5_600 — this exists purely so the fix cannot
        # reduce coverage in a case the sample missed.
        if not rows and QUERY_MODE != "legacy":
            legacy = _lucene_query(query, mode="legacy")
            if legacy and legacy != lucene:
                rows = self._query(legacy, top_k)
        return rows
