"""Backfill Chunk.heading_path / heading_ids: title and clause id of every ancestor heading of a chunk.

    venv/bin/python -m kg_builder.enrich_heading_path --json-dir 3GPP_JSON_DOC/processed_json_v6 [--apply] [--write-json]

--apply      sets the properties on the live KG
--write-json adds section_path / section_path_ids to every chunk of the JSON files themselves, the
             fields the chunker now emits and KGBuilder._create_chunks loads, so a rebuild from a
             corpus chunked before them keeps the headings (atomic write, other fields untouched)

The chunker drops a heading that has no body of its own (create_chunks skips empty sections), so
the title that names a procedure is often absent from the KG: TS 23.502 4.3.2.2 "UE Requested PDU
Session Establishment" exists only as the heading above 4.3.2.2.1 "Non-roaming and Roaming with
Local Breakout", and 4.9.1.2 "Xn based inter NG-RAN handover" only above its three variants.
Retrieval anchored on a procedure name then cannot find the procedure (flow-aggregation pilot,
17/10: root found for 7/33 questions). This re-reads each spec's DOCX (metadata.file_path of its
JSON) with the same parser, keeps every heading, and stores the ancestor titles on each chunk.
Same rule as the chunker (create_chunks: titles of the ancestor clauses that exist as headings).
Dry run by default; Chunk content and ids are untouched.
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "document_processing"))

PART_RE = re.compile(r"(p\d+|~\d+)$")


def _headings(json_path: str, write_json: bool = False):
    """(spec_id, {section_id: title}, [(chunk_id, section_id)]) for one spec; with write_json the
    ancestor fields are also written into the JSON file."""
    from download_and_process_3gpp import ThreeGPPParser  # heavy import, once per worker
    d = json.load(open(json_path, encoding="utf-8"))
    meta = d["metadata"]
    docx = ROOT / meta["file_path"]
    chunks = [(c["chunk_id"], c["section_id"]) for c in d["chunks"]]
    if not docx.is_file():
        return meta["specification_id"], None, chunks
    try:
        sections = ThreeGPPParser().parse_sections(str(docx))
    except Exception as e:  # one unreadable DOCX must not stop the backfill
        print(f"  !! {docx.name}: {e}", file=sys.stderr)
        return meta["specification_id"], None, chunks
    titles = {s.section_id: s.title for s in sections if s.title}
    if write_json:
        for c in d["chunks"]:
            anc = [x for x in ancestors(c["section_id"]) if x in titles]
            c["section_path"] = [titles[x] for x in anc] or None
            c["section_path_ids"] = anc or None
        tmp = Path(json_path).with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        os.replace(tmp, json_path)
    return meta["specification_id"], titles, chunks


def ancestors(section_id: str):
    sid = PART_RE.sub("", section_id)
    parts = sid.split(".")
    return [".".join(parts[:k]) for k in range(1, len(parts))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--write-json", action="store_true")
    ap.add_argument("--only", nargs="*", help="spec ids, e.g. ts_23_502")
    ap.add_argument("--workers", type=int, default=int(os.getenv("CHUNK_WORKERS", "0")) or 6)
    a = ap.parse_args()
    files = sorted(Path(a.json_dir).glob("*.json"))
    if a.only:
        files = [f for f in files if f.stem in a.only]
    rows, n_missing_docx, n_path = [], 0, 0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(_headings, str(f), a.write_json) for f in files]
        for i, fut in enumerate(as_completed(futs), 1):
            spec, titles, chunks = fut.result()
            if titles is None:
                n_missing_docx += 1
                continue
            for cid, sid in chunks:
                anc = [x for x in ancestors(sid) if x in titles]
                n_path += bool(anc)
                rows.append({"i": cid, "p": [titles[x] for x in anc], "s": anc})
            if i % 100 == 0:
                print(f"  {i}/{len(files)} specs")
    print(f"{len(files)} specs, {n_missing_docx} without a readable DOCX, {len(rows)} chunks, "
          f"{n_path} with at least one ancestor title")
    if not a.apply:
        for r in rows[:5]:
            print("  e.g.", r)
        return
    from dotenv import load_dotenv
    from neo4j import GraphDatabase
    load_dotenv(ROOT / ".env")
    drv = GraphDatabase.driver(os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
                               auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password")))
    with drv.session() as s:
        for k in range(0, len(rows), 2000):
            s.run("UNWIND $rows AS r MATCH (c:Chunk {chunk_id: r.i}) SET c.heading_path = r.p, c.heading_ids = r.s", rows=rows[k:k + 2000])
    drv.close()
    print("written")


if __name__ == "__main__":
    main()
