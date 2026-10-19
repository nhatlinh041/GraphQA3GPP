"""
Backfill P2 (Parameter, Step) + P3 (Message) onto an existing KG WITHOUT a
rebuild — calls the new `KGBuilder` methods directly (`_create_parameters`,
`_create_procedure_steps`, `_create_messages`), avoiding logic duplication with
`load_json_dir` (one implementation shared by both build-from-scratch and
backfill, the same way `enrich_terms.py` reuses `TermExtractor`).

Use when: the KG is already built (Document/Chunk/Term/MENTIONS/ServiceOperation/
CO_OCCURS_WITH already present) and you only want to add P2/P3 without waiting for
a full rebuild (~1800 files; rebuilding all Chunk/REFERENCES_*/Term takes a long
time). After the next `npm run rebuild-kg full`, P2/P3 come for free (integrated
into `load_json_dir`) — this script is only needed for a one-off backfill on a
running KG.

Run:
    venv/bin/python -m kg_builder.enrich_procedure_kg
    venv/bin/python -m kg_builder.enrich_procedure_kg --json-dir 3GPP_JSON_DOC/processed_json_v6
    venv/bin/python -m kg_builder.enrich_procedure_kg --skip-parameters --skip-steps
"""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from kg_builder.builder import KGBuilder

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DEFAULT_JSON_DIR = "3GPP_JSON_DOC/processed_json_v6"


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill Parameter/Step/Message (P2+P3) onto an existing KG")
    ap.add_argument("--json-dir", default=DEFAULT_JSON_DIR)
    ap.add_argument("--skip-parameters", action="store_true")
    ap.add_argument("--skip-steps", action="store_true")
    ap.add_argument("--skip-messages", action="store_true")
    args = ap.parse_args()

    uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pwd = os.getenv("NEO4J_PASSWORD", "password")
    builder = KGBuilder(uri, user, pwd)

    builder.setup_schema()  # ensure Parameter/Step/Message constraints exist (idempotent)

    json_dir = Path(args.json_dir)
    files = sorted(json_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No JSON files in {json_dir}")
    _documents, chunks = builder._load_json_files(files)  # noqa: SLF001 — same package
    print(f"[enrich] Loaded {len(chunks)} chunks from {len(files)} files")

    if not args.skip_parameters:
        n = builder._create_parameters(chunks)  # noqa: SLF001
        print(f"[enrich] Parameter nodes: {n}")
    if not args.skip_steps:
        n = builder._create_procedure_steps(chunks)  # noqa: SLF001
        print(f"[enrich] Step nodes: {n}")
    if not args.skip_messages:
        n = builder._create_messages(chunks)  # noqa: SLF001
        print(f"[enrich] Message nodes: {n}")

    builder.close()


if __name__ == "__main__":
    main()
