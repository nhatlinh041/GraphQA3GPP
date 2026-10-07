"""
Enrich KG with `ServiceOperation` nodes + edges `(so)-[:PROVIDED_BY]->(t:Term)` +
`(c:Chunk)-[:DESCRIBES_OPERATION]->(so)` — extracted from the 3GPP SBI
(Service-Based Interface) pattern `N<nf>_<Service>_<Operation>` (e.g. `Nudm_SDM_Get`,
`Nausf_UEAuthentication_Authenticate`) appearing in chunk content.

No re-chunk, no rebuild — reads already-processed JSON directly, purely additive
(MERGE idempotent). Revert with:
    MATCH (so:ServiceOperation) DETACH DELETE so

Why: the KG currently has 0 Term↔Term (network function) relations — the graph
branch only fetches extra Chunks for RRF, no real multi-hop traversal. Service
operations are self-describing (the name embeds the providing NF) and very
regular in SBI specs (23.5xx, 29.5xx...) — measured ~900 distinct tokens, 60+
NF-prefixes, thousands of occurrences across the corpus. Enables questions like
"which operations does UDM provide?" that vector-only cannot answer.
See docs/de_xuat_cai_thien_kg_quan_he.md section P0.

Noise filter: keep only operations whose NF-prefix matches a `Term.abbreviation`
already in the KG (same as how enrich_mentions filters key_terms via
Term.abbreviation) — drops false matches the regex catches (random CamelCase_Case
phrases that aren't real NFs).

Run:
    venv/bin/python -m kg_builder.enrich_service_ops --dry-run
    venv/bin/python -m kg_builder.enrich_service_ops
    venv/bin/python -m kg_builder.enrich_service_ops --json-dir 3GPP_JSON_DOC/processed_json_v6
"""
import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from tqdm import tqdm

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DEFAULT_JSON_DIR = "3GPP_JSON_DOC/processed_json_v6"
WRITE_BATCH_SIZE = 5000

# Service-Based Interface operation name: N<nf-lowercase>_<Service-PascalCase>_<Operation-PascalCase>
# e.g. Nudm_SDM_Get, Nausf_UEAuthentication_Authenticate, Nnwdaf_AnalyticsSubscription_Subscribe.
# Service/Operation must start uppercase — cuts false-positives from ordinary phrases.
SVC_OP_RE = re.compile(r"\bN[a-z][a-z0-9]*_[A-Z][A-Za-z0-9]*_[A-Z][A-Za-z0-9]*\b")


def _parse_op(name: str) -> tuple[str, str, str]:
    """'Nudm_SDM_Get' -> ('UDM', 'SDM', 'Get'). maxsplit=2 keeps the operation
    intact if it happens to contain an extra '_' (rare for this pattern)."""
    nf, service, operation = name.split("_", 2)
    return nf[1:].upper(), service, operation


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create ServiceOperation nodes + PROVIDED_BY/DESCRIBES_OPERATION from the SBI pattern"
    )
    ap.add_argument("--json-dir", default=DEFAULT_JSON_DIR)
    ap.add_argument("--dry-run", action="store_true", help="Print stats only, no writes")
    args = ap.parse_args()

    json_dir = Path(args.json_dir)
    files = sorted(json_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No JSON files in {json_dir}")

    uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pwd = os.getenv("NEO4J_PASSWORD", "password")
    driver = GraphDatabase.driver(uri, auth=(user, pwd))

    def run(cypher, **params):
        with driver.session() as s:
            return s.run(cypher, **params).data()

    # Only create PROVIDED_BY to Terms that already exist.
    term_set = {r["a"] for r in run("MATCH (t:Term) RETURN t.abbreviation AS a")}
    print(f"[svcop] Term abbreviations trong KG: {len(term_set)}")

    op_meta: dict[str, dict] = {}   # name -> {nf, service, operation}
    op_df: Counter = Counter()      # name -> distinct chunk count (document frequency)
    chunk_edges: list[dict] = []    # [{chunk_id, name}]
    n_chunks_scanned = 0
    dropped_no_term: Counter = Counter()

    for jf in tqdm(files, desc="[svcop] Scanning JSON"):
        try:
            data = json.load(open(jf, encoding="utf-8"))
        except Exception as e:
            print(f"[skip] {jf.name}: {e}")
            continue
        for chunk in data.get("chunks", []):
            n_chunks_scanned += 1
            content = chunk.get("content") or ""
            found = set(SVC_OP_RE.findall(content))
            if not found:
                continue
            chunk_id = chunk.get("chunk_id")
            for name in found:
                nf, service, operation = _parse_op(name)
                if nf not in term_set:
                    dropped_no_term[nf] += 1
                    continue
                op_meta.setdefault(name, {"nf": nf, "service": service, "operation": operation})
                op_df[name] += 1
                chunk_edges.append({"chunk_id": chunk_id, "name": name})

    print(f"[svcop] Chunks scanned: {n_chunks_scanned}")
    print(f"[svcop] ServiceOperation distinct (NF-prefix matching Term): {len(op_meta)}")
    print(f"[svcop] DESCRIBES_OPERATION edges to create: {len(chunk_edges)}")
    if dropped_no_term:
        print(
            f"[svcop] Dropped {sum(dropped_no_term.values())} matches (NF-prefix matched no Term) "
            f"— top: {dropped_no_term.most_common(10)}"
        )

    if args.dry_run:
        print("[svcop] --dry-run: nothing written.")
        driver.close()
        return

    # Small count — single batch.
    op_rows = [{"name": name, **meta, "df": op_df[name]} for name, meta in op_meta.items()]
    run(
        """
        UNWIND $rows AS row
        MERGE (so:ServiceOperation {name: row.name})
        SET so.nf_prefix = row.nf, so.service = row.service, so.operation = row.operation, so.df = row.df
        WITH so, row
        MATCH (t:Term {abbreviation: row.nf})
        MERGE (so)-[:PROVIDED_BY]->(t)
        """,
        rows=op_rows,
    )
    print(f"[svcop] ServiceOperation nodes + PROVIDED_BY: {len(op_rows)}")

    # MATCH (not MERGE) on chunk — if chunk_id doesn't exist (spec not yet built
    # into the KG) the row is safely skipped.
    written = 0
    for i in range(0, len(chunk_edges), WRITE_BATCH_SIZE):
        batch = chunk_edges[i : i + WRITE_BATCH_SIZE]
        run(
            """
            UNWIND $rows AS row
            MATCH (c:Chunk {chunk_id: row.chunk_id})
            MATCH (so:ServiceOperation {name: row.name})
            MERGE (c)-[:DESCRIBES_OPERATION]->(so)
            """,
            rows=batch,
        )
        written += len(batch)
        print(f"[svcop] {written}/{len(chunk_edges)}", end="\r")

    total_so = run("MATCH (so:ServiceOperation) RETURN count(so) AS n")[0]["n"]
    total_pb = run("MATCH ()-[r:PROVIDED_BY]->() RETURN count(r) AS n")[0]["n"]
    total_do = run("MATCH ()-[r:DESCRIBES_OPERATION]->() RETURN count(r) AS n")[0]["n"]
    print(f"\n[svcop] Done — ServiceOperation: {total_so}, PROVIDED_BY: {total_pb}, DESCRIBES_OPERATION: {total_do}")
    driver.close()


if __name__ == "__main__":
    main()
