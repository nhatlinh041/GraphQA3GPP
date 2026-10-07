"""
Backfill Layer C (Concept / StandardizedValue) onto an existing KG WITHOUT a
rebuild — calls `KGBuilder._create_standardized_values` directly, the same
single-implementation pattern as `enrich_procedure_kg.py` (no duplicated
extraction logic between build-from-scratch and backfill).

Use when: the KG is already built and you only want the standardized-value
layer (SST/5QI/enumerations/cause tables → Concept-HAS_VALUE->StandardizedValue
with DEFINED_IN_TABLE provenance and DENOTES/ABOUT bridges to Term). After the
next full rebuild this layer comes for free (integrated into `load_json_dir`).

Run:
    .venv/bin/python -m kg_builder.enrich_standardized_values
    .venv/bin/python -m kg_builder.enrich_standardized_values --json-dir 3GPP_JSON_DOC/processed_json_v6
    .venv/bin/python -m kg_builder.enrich_standardized_values --dry-run
"""
import argparse
import collections
import os
from pathlib import Path

from dotenv import load_dotenv

from kg_builder.builder import KGBuilder

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DEFAULT_JSON_DIR = "3GPP_JSON_DOC/processed_json_v6"


def dry_run(builder: KGBuilder, chunks: list) -> None:
    """Run the EXACT collection pipeline the write path uses (same
    classification, same row filters, same gates) without writing to Neo4j —
    the audit must sign off on the dataset that actually gets ingested."""
    rows, concept_meta = builder.collect_standardized_value_rows(chunks)

    kind_counter = collections.Counter(m["kind"] for m in concept_meta.values())
    concept_counter = collections.Counter()
    spec_counter = collections.Counter()
    for r in rows:
        concept_counter[r["concept"]] += 1
        spec_counter[r["spec_id"]] += 1
    print(f"[dry-run] concepts: {len(concept_meta)} (kinds {dict(kind_counter)}), "
          f"rows to ingest: {len(rows)}")
    print("[dry-run] top 20 concepts by row count:")
    for name, n in concept_counter.most_common(20):
        print(f"    {name}: {n} rows")
    print("[dry-run] top specs:", spec_counter.most_common(10))
    print("[dry-run] samples:")
    for r in rows[:15]:
        print(f"    {r['value_id']:44} {r['concept']:24} {r['name'][:28]:28} "
              f"code={r['code'][:16]}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Backfill Concept/StandardizedValue (Layer C) onto an existing KG")
    ap.add_argument("--json-dir", default=DEFAULT_JSON_DIR)
    ap.add_argument("--dry-run", action="store_true",
                    help="classify + print stats only, write nothing")
    args = ap.parse_args()

    uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pwd = os.getenv("NEO4J_PASSWORD", "password")
    builder = KGBuilder(uri, user, pwd)

    json_dir = Path(args.json_dir)
    files = sorted(json_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No JSON files in {json_dir}")
    _documents, chunks = builder._load_json_files(files)  # noqa: SLF001 — same package
    print(f"[enrich] Loaded {len(chunks)} chunks from {len(files)} files")

    if args.dry_run:
        dry_run(builder, chunks)
        return

    builder.setup_schema()  # ensure Concept/StandardizedValue constraints exist (idempotent)
    n = builder._create_standardized_values(chunks)  # noqa: SLF001
    print(f"[enrich] StandardizedValue rows written: {n}")
    builder.validate()


if __name__ == "__main__":
    main()
