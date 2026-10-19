"""Re-extract cross references with the current chunker rules and rebuild REFERENCES_CHUNK.

    venv/bin/python -m kg_builder.enrich_references --json-dir 3GPP_JSON_DOC/processed_json_v6            # dry run
    venv/bin/python -m kg_builder.enrich_references --json-dir 3GPP_JSON_DOC/processed_json_v6 --write-json
    venv/bin/python -m kg_builder.enrich_references --json-dir 3GPP_JSON_DOC/processed_json_v6 --apply

Two fixes of 18/10 (flow-aggregation pilot: 27 gold clauses cited by a flow yet unreachable):
- extraction (document_processing ThreeGPPParser.extract_cross_references) now captures "Annex A.2",
  letter-suffixed ids ("6.1.4.1a" was cut to 6.1.4.1) and every id of "clauses X and Y";
- matching (KGBuilder._match_ref_to_chunk) links a citation of a heading-only clause to all its
  descendants instead of the first one.
--write-json rewrites only `cross_references` of each chunk (atomic); --apply deletes every
REFERENCES_CHUNK edge and recreates them from the JSON through the builder's own method, so a full
rebuild gives the same edges. REFERENCES_SPEC is re-merged as well (idempotent).
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "document_processing"))


def _rewrite(json_path: str, write: bool) -> tuple[int, int, int]:
    from download_and_process_3gpp import ThreeGPPParser
    d = json.load(open(json_path, encoding="utf-8"))
    spec = d["metadata"]["specification_id"]
    parser = ThreeGPPParser()
    old_n = new_n = changed = 0
    for c in d["chunks"]:
        old = c.get("cross_references") or {}
        new = parser.extract_cross_references(c.get("content") or "", spec)
        old_n += len(old.get("internal", [])) + len(old.get("external", []))
        new_n += len(new["internal"]) + len(new["external"])
        if new != old:
            changed += 1
            c["cross_references"] = new
    if write and changed:
        tmp = Path(json_path).with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        os.replace(tmp, json_path)
    return old_n, new_n, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", required=True)
    ap.add_argument("--write-json", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    files = sorted(Path(a.json_dir).glob("*.json"))
    if not a.apply:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            res = list(pool.map(_rewrite, map(str, files), [a.write_json] * len(files)))
        print(f"{len(files)} specs: refs {sum(r[0] for r in res)} -> {sum(r[1] for r in res)}, "
              f"chunks changed {sum(r[2] for r in res)}" + (" (written)" if a.write_json else " (dry run)"))
        return
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from kg_builder.builder import KGBuilder
    kg = KGBuilder()
    _, chunks = kg._load_json_files(files)
    with kg._driver.session() as s:
        before = s.run("MATCH ()-[r:REFERENCES_CHUNK]->() RETURN count(r) AS n").single()["n"]
        while s.run("MATCH ()-[r:REFERENCES_CHUNK]->() WITH r LIMIT 50000 DELETE r RETURN count(*) AS n").single()["n"]:
            pass
    n = kg._create_references_chunk_edges(chunks)
    kg._create_references_spec_edges(chunks)
    with kg._driver.session() as s:
        after = s.run("MATCH ()-[r:REFERENCES_CHUNK]->() RETURN count(r) AS n").single()["n"]
    print(f"REFERENCES_CHUNK {before} -> {after} ({n} rows merged)")


if __name__ == "__main__":
    main()
