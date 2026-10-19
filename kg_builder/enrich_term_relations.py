"""
Enrich KG with Term↔Term relations — 2 independent parts, each toggleable:

1. `Term.semantic_type = 'network_function'` — inferred from the PROVIDED_BY
   edge created by `enrich_service_ops.py` (a Term providing at least 1
   ServiceOperation is certainly a network function). Do NOT guess 'interface'
   or other types by heuristic — per kg_thay_doi.md, interface identifiers (N1,
   N6, S1, Uu...) do NOT exist as Term.abbreviation in this KG, so labelling a
   Term 'interface' is always empty. Precision-first: only label with direct
   evidence, leave the rest blank (do not force 'generic').

2. `(Term)-[:CO_OCCURS_WITH {weight}]->(Term)` — inferred directly from
   existing MENTIONS (2 Terms MENTIONS the same Chunk), no JSON re-read.
   `weight` = number of distinct chunks where both terms are mentioned.
   Undirected edge, stored one-way (t1.abbreviation < t2.abbreviation) to
   avoid duplicates.

No rebuild, purely additive (MERGE idempotent). Revert:
    MATCH ()-[r:CO_OCCURS_WITH]->() DELETE r
    MATCH (t:Term) REMOVE t.semantic_type

Why needed: the KG currently has 0 relations between Terms — no way to walk
the graph "from AMF to related entities" without going through an intermediate
Chunk. CO_OCCURS_WITH opens real multi-hop, reusing already-built MENTIONS data
(cheap — 1 aggregate query, no JSON scan). See
docs/de_xuat_cai_thien_kg_quan_he.md section P1.

Run:
    venv/bin/python -m kg_builder.enrich_term_relations --dry-run
    venv/bin/python -m kg_builder.enrich_term_relations
    venv/bin/python -m kg_builder.enrich_term_relations --min-weight 3
    venv/bin/python -m kg_builder.enrich_term_relations --skip-semantic-type
    venv/bin/python -m kg_builder.enrich_term_relations --skip-cooccur
"""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def main() -> None:
    ap = argparse.ArgumentParser(description="Create Term.semantic_type + (Term)-[:CO_OCCURS_WITH]->(Term)")
    ap.add_argument("--min-weight", type=int, default=3, help="Drop Term pairs co-occurring in < N chunks (default 3)")
    ap.add_argument("--dry-run", action="store_true", help="Only print stats, no writes")
    ap.add_argument("--skip-semantic-type", action="store_true")
    ap.add_argument("--skip-cooccur", action="store_true")
    args = ap.parse_args()

    uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pwd = os.getenv("NEO4J_PASSWORD", "password")
    driver = GraphDatabase.driver(uri, auth=(user, pwd))

    def run(cypher, **params):
        with driver.session() as s:
            return s.run(cypher, **params).data()

    if not args.skip_semantic_type:
        n_candidates = run(
            "MATCH (t:Term)<-[:PROVIDED_BY]-(:ServiceOperation) "
            "WHERE t.semantic_type IS NULL "
            "RETURN count(DISTINCT t) AS n"
        )[0]["n"]
        print(f"[relations] Terms to set semantic_type='network_function': {n_candidates}")
        if n_candidates == 0:
            print(
                "[relations] No candidates — run `enrich_service_ops.py` first "
                "(needs PROVIDED_BY edge), or semantic_type is already fully set."
            )
        if not args.dry_run and n_candidates > 0:
            run(
                "MATCH (t:Term)<-[:PROVIDED_BY]-(:ServiceOperation) "
                "SET t.semantic_type = 'network_function'"
            )
            total = run(
                "MATCH (t:Term {semantic_type: 'network_function'}) RETURN count(t) AS n"
            )[0]["n"]
            print(f"[relations] Done — Term.semantic_type='network_function': {total}")

    if not args.skip_cooccur:
        # Read-only pre-count — same shape as the write query below so the
        # displayed number exactly matches the edges that will be created.
        n_pairs = run(
            """
            MATCH (t1:Term)<-[:MENTIONS]-(c:Chunk)-[:MENTIONS]->(t2:Term)
            WHERE t1.abbreviation < t2.abbreviation
            WITH t1, t2, count(DISTINCT c) AS weight
            WHERE weight >= $min_weight
            RETURN count(*) AS n
            """,
            min_weight=args.min_weight,
        )[0]["n"]
        print(f"[relations] CO_OCCURS_WITH edges to create (weight >= {args.min_weight}): {n_pairs}")

        if args.dry_run:
            print("[relations] --dry-run: nothing written.")
            driver.close()
            return

        run(
            """
            MATCH (t1:Term)<-[:MENTIONS]-(c:Chunk)-[:MENTIONS]->(t2:Term)
            WHERE t1.abbreviation < t2.abbreviation
            WITH t1, t2, count(DISTINCT c) AS weight
            WHERE weight >= $min_weight
            MERGE (t1)-[r:CO_OCCURS_WITH]->(t2)
            SET r.weight = weight
            """,
            min_weight=args.min_weight,
        )
        total = run("MATCH ()-[r:CO_OCCURS_WITH]->() RETURN count(r) AS n")[0]["n"]
        print(f"[relations] Done — CO_OCCURS_WITH edges in KG: {total}")
    elif args.dry_run:
        print("[relations] --dry-run: nothing written.")

    driver.close()


if __name__ == "__main__":
    main()
