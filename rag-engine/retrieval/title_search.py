"""
Section-title search — a ranked replacement for Pattern B's hand-written regex.

The graph branch already treats `Chunk.section_title` as legitimate graph-side
signal (Pattern B does `c.section_title =~ '(?i).*\\bN6\\b.*'`), but it reaches
that field through a regex the Cypher-gen LLM writes, which is brittle in both
directions: it misses titles that phrase the same clause differently, and it
matches superficially similar titles in an unrelated spec. Measured on
kg_crossspec_150, 37 of the 43 questions whose Cypher returned candidates that
the cross-encoder then rejected had candidates from specs that share NOTHING
with the gold ("S1 Setup procedure" -> ts_25_912, a UMTS study report).

This module queries the same field through a dedicated Lucene index instead, so
the match is ranked rather than boolean and robust to wording. Titles average
4.1 tokens, so a title-only index is a genuinely different retriever from
`chunk_fulltext` (content + title) — not a duplicate of the BM25 branch.

Measured Recall@5 of A_kg_only (v14/v10 chunk lists vs. an RRF merge with this
retriever's top-20):

    kg_crossspec_150   0.037 -> 0.167
    wh_kg_v5_600       0.301 -> 0.512   (regression check, no loss)

Merging beats replacing on both sets, and beats running only as a fallback when
Cypher comes up empty (0.139): title search also improves the questions where
Cypher DID return something (0.089 -> 0.128 on the 150-set).
"""
import os
import re

from neo4j import GraphDatabase

# Same tokenisation contract as bm25_search: reduce the question to bare terms so
# Lucene's query parser never sees an operator, but keep "_" because the index
# analyzer treats it as a joiner (UAX#29 ExtendNumLet). See bm25_search for the
# measurement behind that choice.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*")

# Question words carry no title signal and, over a 4-token field, a single common
# term dominates BM25's length normalisation. Kept deliberately short — this is a
# stop list for interrogatives, not a general-purpose one.
_STOP = frozenset(
    "the a an of for in on to and or is are was were what which when where who whom "
    "how does do did this that these those with by from at as its it be been being "
    "must can could should would may might each case several more than there here "
    "any some all both".split()
)

TITLE_INDEX_NAME = "chunk_title_fulltext"

# Single source of truth; kg_builder.CYPHER_INDEXES carries the build-time copy.
TITLE_FULLTEXT_DDL = (
    f"CREATE FULLTEXT INDEX {TITLE_INDEX_NAME} IF NOT EXISTS "
    "FOR (c:Chunk) ON EACH [c.section_title]"
)

# Off-switch for ablation. Default ON — this is a measured improvement, and the
# paper needs the "with / without" row.
TITLE_SEARCH_OFF = os.getenv("TITLE_SEARCH_OFF", "0") == "1"
TOP_K_TITLE = int(os.getenv("TOP_K_TITLE", "20"))
RRF_K = int(os.getenv("TITLE_RRF_K", "60"))


def title_query(text: str) -> str:
    """Question -> space-joined Lucene terms, interrogatives dropped."""
    return " ".join(t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP)


def rrf_merge(graph_chunks: list[dict], title_chunks: list[dict], k: int = RRF_K) -> list[dict]:
    """Reciprocal-rank fusion of the Cypher result and the title result.

    Runs AFTER _normalise_graph_scores in the orchestrator, so the output score is
    already on the [0,1] scale the rerank blend and the flat-score guard expect —
    merging before the raw-score filter would let GRAPH_MIN_SCORE drop every title
    hit, since a Lucene score is not a Pattern-A confidence.
    """
    scores: dict[str, float] = {}
    payload: dict[str, dict] = {}
    for source in (graph_chunks, title_chunks):
        for rank, c in enumerate(source):
            cid = c.get("chunk_id")
            if not cid:
                continue
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            # Keep the Cypher row when a chunk is in both: it carries the pattern's
            # own metadata (n_steps, n_via_op) that downstream bias functions read.
            payload.setdefault(cid, c)
    if not scores:
        return []
    top = max(scores.values())
    out = []
    for cid, s in sorted(scores.items(), key=lambda kv: -kv[1]):
        c = dict(payload[cid])
        c["score"] = s / top
        out.append(c)
    return out


class TitleSearcher:
    def __init__(self, uri: str, user: str, password: str, index_name: str = TITLE_INDEX_NAME):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._index_name = index_name

    def close(self) -> None:
        self._driver.close()

    def ensure_index(self) -> None:
        """Create the title index if the KG was built before this feature."""
        with self._driver.session() as session:
            session.run(TITLE_FULLTEXT_DDL.replace(TITLE_INDEX_NAME, self._index_name))

    def search(self, query: str, top_k: int = TOP_K_TITLE) -> list[dict]:
        lucene = title_query(query)
        if not lucene:
            return []
        cypher = """
        CALL db.index.fulltext.queryNodes($index_name, $search, {limit: $top_k})
        YIELD node AS chunk, score
        RETURN
            chunk.chunk_id      AS chunk_id,
            chunk.content       AS content,
            chunk.spec_id       AS spec_id,
            chunk.section_title AS section,
            score
        ORDER BY score DESC
        """
        with self._driver.session() as session:
            rows = session.run(
                cypher, index_name=self._index_name, search=lucene, top_k=top_k
            )
            return [dict(r) for r in rows]
