"""
Serialize the KG neighbourhood of retrieved chunks into text triples.

For the paper's Study-Ablation sheet, the "KG-only" row supplies the LLM with
*KG triples* rather than chunk prose. The normal pipeline always feeds chunk
text (even in kg_only mode the graph search returns Chunk nodes), so this module
provides the triples-as-evidence variant, gated by env KG_EVIDENCE=triples.

Edges emitted per chunk (schema in docs/kg_schema.md):
  (Chunk)-[:DESCRIBES_OPERATION]->(ServiceOperation)-[:PROVIDED_BY]->(Term)
  (Chunk)-[:HAS_STEP]->(Step)-[:INVOLVES]->(Term)
  (Chunk)-[:DESCRIBES_MESSAGE]->(Message)
  (Parameter)-[:DEFINED_IN_TABLE]->(Chunk)
  (Chunk)-[:MENTIONS]->(Term)          (capped — the noisiest edge)
  (Chunk)-[:REFERENCES_SPEC]->(spec)
Each chunk is titled with its own [spec §section] so provenance survives.
"""
from neo4j import GraphDatabase  # noqa: F401  (type hint only; driver passed in)

_CYPHER = """
UNWIND $ids AS cid
MATCH (c:Chunk {chunk_id: cid})
OPTIONAL MATCH (c)-[:DESCRIBES_OPERATION]->(so:ServiceOperation)-[:PROVIDED_BY]->(nf:Term)
WITH c, cid, collect(DISTINCT so.name + ' PROVIDED_BY ' + nf.abbreviation) AS ops
OPTIONAL MATCH (c)-[:HAS_STEP]->(st:Step)-[:INVOLVES]->(t:Term)
WITH c, cid, ops, collect(DISTINCT st.step_id + ' INVOLVES ' + t.abbreviation)[0..$per] AS steps
OPTIONAL MATCH (c)-[:DESCRIBES_MESSAGE]->(m:Message)
WITH c, cid, ops, steps, collect(DISTINCT m.name) AS msgs
OPTIONAL MATCH (p:Parameter)-[:DEFINED_IN_TABLE]->(c)
WITH c, cid, ops, steps, msgs, collect(DISTINCT p.name)[0..$per] AS params
OPTIONAL MATCH (c)-[:MENTIONS]->(mt:Term)
WITH c, cid, ops, steps, msgs, params, collect(DISTINCT mt.abbreviation)[0..$per] AS mentions
OPTIONAL MATCH (c)-[:REFERENCES_SPEC]->(d:Document)
RETURN cid AS id, c.spec_id AS spec, c.section_title AS sec,
       ops, steps, msgs, params, mentions,
       collect(DISTINCT d.spec_id) AS refs
"""


def serialize_triples(driver, chunk_ids: list[str], per_edge_cap: int = 12,
                      max_chunks: int = 10) -> str:
    """Return a text block of KG triples for the top `max_chunks` retrieved chunks,
    in rank order. Empty string if nothing to serialize."""
    if not chunk_ids:
        return ""
    ids = chunk_ids[:max_chunks]
    rows = {}
    with driver.session() as s:
        for r in s.run(_CYPHER, ids=ids, per=per_edge_cap):
            rows[r["id"]] = r
    blocks = []
    for cid in ids:                       # preserve retrieval rank order
        r = rows.get(cid)
        if not r:
            continue
        lines = []
        for op in (r["ops"] or []):
            lines.append(f"  {op}")
        for stp in (r["steps"] or []):
            lines.append(f"  {stp}")
        for msg in (r["msgs"] or []):
            lines.append(f"  {cid} DESCRIBES_MESSAGE {msg}")
        for pm in (r["params"] or []):
            lines.append(f"  Parameter[{pm}] DEFINED_IN_TABLE {cid}")
        if r["mentions"]:
            lines.append(f"  {cid} MENTIONS " + ", ".join(r["mentions"]))
        for ref in (r["refs"] or []):
            lines.append(f"  {r['spec']} REFERENCES_SPEC {ref}")
        header = f"[{r['spec']} §{r['sec']}]"
        blocks.append(header + "\n" + ("\n".join(lines) if lines else "  (no graph relations)"))
    return "\n\n".join(blocks)
