"""
Multi-hop graph traversal — follows REFERENCES_CHUNK edges across specs.
Implements 5 hop patterns from the research plan.

REFERENCES_CHUNK covers BOTH internal (same-spec) and external (cross-spec)
section references. Edge property `is_external` distinguishes the two groups;
variable-length traversal `[:REFERENCES_CHUNK*1..3]` crosses both uniformly.
To restrict to cross-spec only, filter `WHERE r.is_external = true` (add
`r.confidence >= 0.7` for high-quality refs).
"""
from neo4j import GraphDatabase


# Pattern 1: Term → Chunk (via source_specs) → referenced Chunk (1 hop)
# NOTE: gold pattern uses `c.spec_id IN t.source_specs` instead of DEFINED_IN→Chunk
# (DEFINED_IN points Term→Document, not Term→Chunk).
PATTERN_1_HOP = """
MATCH (t:Term {abbreviation: $seed})
WITH t LIMIT 1
MATCH (c:Chunk) WHERE c.spec_id IN t.source_specs
MATCH (c)-[:REFERENCES_CHUNK]->(related:Chunk)
RETURN related.chunk_id AS chunk_id, related.content AS content,
       related.spec_id AS spec_id, related.section_id AS section,
       0.8 AS score, 1 AS hops
LIMIT $top_k
"""

# Pattern 2: Cross-spec via REFERENCES_SPEC (2 hops)
# Seed is a spec_id (e.g. 'ts_23_501'), not a term — self-fails if seed is an abbreviation.
PATTERN_2_CROSS_SPEC = """
MATCH (c1:Chunk {spec_id: $seed})-[:REFERENCES_SPEC]->(d:Document)<-[:CONTAINS]-(c2:Chunk)
RETURN c2.chunk_id AS chunk_id, c2.content AS content,
       c2.spec_id AS spec_id, c2.section_id AS section,
       0.75 AS score, 2 AS hops
LIMIT $top_k
"""

# Pattern 3: Term → Chunk → Chunk → Term (term chain, 2 hops)
# Also uses the source_specs property instead of DEFINED_IN for the Term-Chunk side.
PATTERN_3_TERM_CHAIN = """
MATCH (t1:Term {abbreviation: $seed})
WITH t1 LIMIT 1
MATCH (c1:Chunk) WHERE c1.spec_id IN t1.source_specs
MATCH (c1)-[:REFERENCES_CHUNK]->(c2:Chunk)
MATCH (t2:Term) WHERE c2.spec_id IN t2.source_specs AND t2.abbreviation <> $seed
RETURN c2.chunk_id AS chunk_id, c2.content AS content,
       c2.spec_id AS spec_id, c2.section_id AS section,
       0.7 AS score, 2 AS hops
LIMIT $top_k
"""

# Pattern 4: Subject-guided traversal
# Seed is a Subject name (e.g. 'Lexicon') — silently fails if seed does not match.
PATTERN_4_SUBJECT = """
MATCH (c1:Chunk)-[:HAS_SUBJECT]->(s:Subject {name: $seed})<-[:HAS_SUBJECT]-(c2:Chunk)
WHERE c1.chunk_id <> c2.chunk_id
RETURN c2.chunk_id AS chunk_id, c2.content AS content,
       c2.spec_id AS spec_id, c2.section_id AS section,
       0.65 AS score, 1 AS hops
LIMIT $top_k
"""

# Pattern 5: Variable-length path (up to 3 hops) via REFERENCES_CHUNK
# Starts from the seed term's chunks (via source_specs).
PATTERN_5_VARIABLE = """
MATCH (t:Term {abbreviation: $seed})
WITH t LIMIT 1
MATCH (start:Chunk) WHERE start.spec_id IN t.source_specs
WITH start LIMIT 5
MATCH path = (start)-[:REFERENCES_CHUNK*1..3]->(end:Chunk)
RETURN end.chunk_id AS chunk_id, end.content AS content,
       end.spec_id AS spec_id, end.section_id AS section,
       0.6 AS score, length(path) AS hops
LIMIT $top_k
"""

ALL_PATTERNS = [
    PATTERN_1_HOP,
    PATTERN_2_CROSS_SPEC,
    PATTERN_3_TERM_CHAIN,
    PATTERN_4_SUBJECT,
    PATTERN_5_VARIABLE,
]


class MultiHopSearcher:
    def __init__(self, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def search(self, seeds: list[str], top_k: int = 6) -> list[dict]:
        """Run all 5 hop patterns for each seed term, return merged results."""
        seen: set[str] = set()
        results: list[dict] = []

        with self._driver.session() as session:
            for seed in seeds:
                for pattern in ALL_PATTERNS:
                    try:
                        rows = session.run(pattern, seed=seed, top_k=top_k)
                        for row in rows:
                            r = dict(row)
                            if r["chunk_id"] not in seen:
                                seen.add(r["chunk_id"])
                                results.append(r)
                    except Exception:
                        # Pattern may not apply if KG edges are missing — skip silently
                        continue

        # Sort by score desc, then by fewest hops
        results.sort(key=lambda x: (-x["score"], x.get("hops", 99)))
        return results[:top_k]
