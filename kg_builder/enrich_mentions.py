"""
Enrich KG with edge `(Chunk)-[:MENTIONS]->(Term)` from `Chunk.key_terms` — NO
re-chunk, NO rebuild. Purely additive (MERGE idempotent), revertible with:
    MATCH ()-[r:MENTIONS]->() DELETE r

Why needed: KG currently has no Chunk→Term edge; retrieval joins Term→Chunk via
`c.spec_id IN t.source_specs` (document-level, low precision). MENTIONS edge gives
a more precise chunk-level anchor + enables multi-hop. See TODO/proposal_kg_edges.md.

Noise filter: only create an edge for a key_term that MATCHES `Term.abbreviation`
AND has document-frequency within band [MIN_DF, MAX_DF] — drops ubiquitous terms
(TS, UE, NOTE, IDF≈0) and overly rare terms (extraction noise). Store `r.df` for
use as a weight (IDF) later.

Run:
    venv/bin/python -m kg_builder.enrich_mentions --dry-run      # print stats only
    venv/bin/python -m kg_builder.enrich_mentions                # create edges
    venv/bin/python -m kg_builder.enrich_mentions --min-df 5 --max-df 5000
"""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load .env at repo root (like enrich_terms / main.py) before reading NEO4J_*
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

WRITE_BATCH_SIZE = 5000


def main() -> None:
    ap = argparse.ArgumentParser(description="Create edge (Chunk)-[:MENTIONS]->(Term)")
    ap.add_argument("--min-df", type=int, default=5, help="Drop rare terms (df < min)")
    ap.add_argument("--max-df", type=int, default=5000, help="Drop ubiquitous terms (df > max)")
    ap.add_argument("--dry-run", action="store_true", help="Print stats only, no writes")
    args = ap.parse_args()

    uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pwd = os.getenv("NEO4J_PASSWORD", "password")
    driver = GraphDatabase.driver(uri, auth=(user, pwd))

    def run(cypher, **params):
        with driver.session() as s:
            return s.run(cypher, **params).data()

    # Valid Term.abbreviation set (only create edges to existing Terms).
    term_set = {r["a"] for r in run("MATCH (t:Term) RETURN t.abbreviation AS a")}
    print(f"[mentions] Term abbreviations: {len(term_set)}")

    df_rows = run(
        "MATCH (c:Chunk) WHERE c.key_terms IS NOT NULL "
        "UNWIND c.key_terms AS kt RETURN kt, count(*) AS df"
    )
    keep = {
        r["kt"]: r["df"]
        for r in df_rows
        if r["kt"] in term_set and args.min_df <= r["df"] <= args.max_df
    }
    dropped_common = sum(1 for r in df_rows if r["kt"] in term_set and r["df"] > args.max_df)
    dropped_rare = sum(1 for r in df_rows if r["kt"] in term_set and r["df"] < args.min_df)
    print(
        f"[mentions] key_terms matching Term & df∈[{args.min_df},{args.max_df}]: {len(keep)} "
        f"(dropped {dropped_common} ubiquitous df>{args.max_df}, {dropped_rare} rare df<{args.min_df})"
    )

    chunk_rows = run(
        "MATCH (c:Chunk) WHERE c.key_terms IS NOT NULL "
        "RETURN c.chunk_id AS id, c.key_terms AS kt"
    )
    rows = [
        {"id": cr["id"], "abbr": kt, "df": keep[kt]}
        for cr in chunk_rows
        for kt in (cr["kt"] or [])
        if kt in keep
    ]
    print(f"[mentions] edges to create: {len(rows)} (over {len(chunk_rows)} chunks)")

    if args.dry_run:
        print("[mentions] --dry-run: nothing written.")
        driver.close()
        return

    # MATCH (t:Term) guarantees edges are only created to existing Terms.
    written = 0
    for i in range(0, len(rows), WRITE_BATCH_SIZE):
        batch = rows[i : i + WRITE_BATCH_SIZE]
        run(
            """
            UNWIND $rows AS row
            MATCH (c:Chunk {chunk_id: row.id})
            MATCH (t:Term {abbreviation: row.abbr})
            MERGE (c)-[r:MENTIONS]->(t)
            SET r.df = row.df
            """,
            rows=batch,
        )
        written += len(batch)
        print(f"[mentions] {written}/{len(rows)}", end="\r")

    total = run("MATCH ()-[r:MENTIONS]->() RETURN count(r) AS n")[0]["n"]
    print(f"\n[mentions] Done — MENTIONS edges in KG: {total}")
    driver.close()


if __name__ == "__main__":
    main()
