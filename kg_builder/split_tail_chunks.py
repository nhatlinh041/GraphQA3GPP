"""
Repair the chunker's tail-swallow on an ALREADY-BUILT KG, without a rebuild.

The bug (see `document_processing/download_and_process_3gpp.py`, heading grammar
block): `parse_sections` only opened a new section on a purely numeric heading, so
every Annex root, every annex sub-clause (A.1, A.7.0) and the change-history table
fell through to "append to the section currently open". The result is that each
document's LAST numbered clause absorbs the whole tail of the document — measured
live: 1 770 chunks carry a change-history table, 234 exceed 100 KB, the largest is
1.57 MB, and `ts_33_501_16.6.3` ("Subscription/unsubscription of NSACF notification
service", 262 KB) contains Annex A→AA including the EAP-TLS flow. Those chunks
poison retrieval twice: they win anchors/vector for content they only *contain*,
and their excerpt reaches the answer under a completely unrelated section citation.

This script cuts the tail back out of the live KG:
  * the head (the clause's own text) stays on the original chunk_id;
  * each Annex and annex sub-clause becomes its own chunk (`ts_33_501_A`,
    `ts_33_501_A.1`, …) so the knowledge is kept, not deleted;
  * the change history and everything after it is dropped;
  * derived nodes/edges of every touched chunk are rebuilt through the SAME
    `KGBuilder._create_*` methods the build pipeline uses (no duplicated logic),
    and embeddings are cleared so `npm run rebuild-kg:embed-only` tops them up.

The heading grammar is imported from the chunker — one source of truth, so a fix
here and a fix at chunking time can never drift.

DRY-RUN BY DEFAULT. Nothing is written without `--apply`.

Known limits of the in-place path (the full re-chunk to a new JSON dir + rebuild
does not have them):
  * `Parameter` / `DEFINED_IN_TABLE` are derived from the JSON `tables` array,
    which the KG does not store — they are left untouched (use `--drop-parameters`
    to delete the stale ones instead of keeping them).
  * `Concept` / `StandardizedValue` come from the same tables — untouched.
  * `CO_OCCURS_WITH` is a global aggregate; re-run `kg_builder.enrich_term_relations`
    afterwards if you care about it.

Run:
    .venv/bin/python -m kg_builder.split_tail_chunks                 # dry-run report
    .venv/bin/python -m kg_builder.split_tail_chunks --limit 20 -v   # inspect a few
    .venv/bin/python -m kg_builder.split_tail_chunks --apply         # write
    REEMBED after: npm run rebuild-kg:embed-only
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from typing import Iterator, Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase

from kg_builder.builder import KGBuilder

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

_CHUNKER_PATH = ROOT / "document_processing" / "download_and_process_3gpp.py"

# Prefilter for the candidate scan. `(normative)` / `(informative)` catch annex
# roots; 'Change history' catches documents whose tail is only the CR table. Kept
# as a server-side CONTAINS so we stream ~1.9k chunks instead of all 183k.
_CANDIDATE_CYPHER = """
MATCH (c:Chunk)
WHERE c.content CONTAINS 'Change history'
   OR c.content CONTAINS '(normative)'
   OR c.content CONTAINS '(informative)'
RETURN c.chunk_id AS chunk_id, c.spec_id AS spec_id, c.section_id AS section_id,
       c.section_title AS section_title, c.content AS content
ORDER BY c.chunk_id
SKIP $skip LIMIT $limit
"""


def _load_chunker():
    """Import the chunker by path — `document_processing/` is a script dir, not a
    package, so a normal import is unavailable."""
    spec = importlib.util.spec_from_file_location("dp_3gpp", _CHUNKER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Block:
    __slots__ = ("section_id", "title", "lines")

    def __init__(self, section_id: str, title: str):
        self.section_id = section_id
        self.title = title
        self.lines: list[str] = []

    @property
    def content(self) -> str:
        return "\n".join(self.lines).strip()


def plan_split(
    section_id: str,
    section_title: str,
    content: str,
    chunker,
    split_letter_clauses: bool = False,
) -> Optional[tuple[str, list[_Block], bool]]:
    """Split one chunk's flat content back into the sections the chunker should
    have produced.

    Returns `(head_content, tail_blocks, dropped_history)` or None when the chunk
    carries no swallowed tail.

    A line opens a new block when it is an annex root, or an annex sub-clause of
    the annex currently open (`A.1` inside `Annex A`). Only ids CONTAINING A
    LETTER are accepted as headings: in flat text a purely numeric "5<tab>…" line
    is far more likely to be a table row than a heading, and the head region must
    not be shredded by a false positive. `--split-letter-clauses` extends the same
    rule outside annexes, recovering the other half of the bug (clause 7A/7B of
    TS 33.501 swallowed by 7.2.1) at a slightly higher false-positive risk.
    """
    lines = content.split("\n")
    head = _Block(section_id, section_title)
    blocks: list[_Block] = [head]
    cur = head
    annex_letter: Optional[str] = None
    dropped_history = False

    for line in lines:
        stripped = line.strip()

        annex = chunker.match_annex_heading(stripped)
        if annex:
            annex_id, annex_title = annex
            if chunker.is_change_history(annex_title):
                dropped_history = True
                break
            annex_letter = annex_id
            cur = _Block(annex_id, annex_title)
            blocks.append(cur)
            continue

        if chunker.is_change_history(stripped):
            dropped_history = True
            break

        sub = chunker.SECTION_HEADING_ALT_RE.match(line)
        if sub:
            sub_id, sub_title = sub.group(1), sub.group(2).strip()
            has_letter = any(ch.isalpha() for ch in sub_id)
            in_this_annex = (
                annex_letter is not None and sub_id.split(".")[0] == annex_letter
            )
            if has_letter and (in_this_annex or (split_letter_clauses and annex_letter is None)):
                cur = _Block(sub_id, sub_title)
                blocks.append(cur)
                continue

        cur.lines.append(line)

    tail = [b for b in blocks[1:] if b.content]
    if not tail and not dropped_history:
        return None
    return head.content, tail, dropped_history


def _to_chunk_dicts(spec_id: str, section_id: str, title: str, content: str,
                    chunker, parser) -> list[dict]:
    """Chunk dicts in the shape `KGBuilder._create_*` expects, honouring the
    chunker's MAX_CHUNK_CHARS cap (same `p2`/`p3` part naming)."""
    out = []
    parts = parser._split_oversized(content, chunker.MAX_CHUNK_CHARS)
    for idx, part in enumerate(parts, start=1):
        if not part.strip():
            continue
        sid = section_id if idx == 1 else f"{section_id}p{idx}"
        ttl = title if idx == 1 else f"{title} (part {idx})"
        out.append({
            "chunk_id": f"{spec_id}_{sid}",
            "_spec_id": spec_id,
            "section_id": sid,
            "section_title": ttl,
            "content": part,
            "chunk_type": parser.classify_content_type(ttl, part),
            "cross_references": parser.extract_cross_references(part, spec_id),
            "content_metadata": {
                "word_count": len(part.split()),
                "complexity_score": parser.compute_complexity(part),
                "key_terms": parser.extract_key_terms(part),
            },
        })
    return out


def _iter_candidates(driver, page: int = 200) -> Iterator[dict]:
    skip = 0
    while True:
        with driver.session(default_access_mode="READ") as s:
            rows = [dict(r) for r in s.run(_CANDIDATE_CYPHER, skip=skip, limit=page)]
        if not rows:
            return
        yield from rows
        skip += len(rows)


# Derived edges/nodes of a repaired chunk describe content it no longer has.
# Parameter is deliberately absent (see module docstring).
_PURGE_CYPHER = """
UNWIND $ids AS cid
MATCH (c:Chunk {chunk_id: cid})
OPTIONAL MATCH (c)-[r:MENTIONS|DESCRIBES_OPERATION|DESCRIBES_MESSAGE|REFERENCES_CHUNK|REFERENCES_SPEC|HAS_SUBJECT]->()
DELETE r
WITH DISTINCT c
OPTIONAL MATCH (c)-[:HAS_STEP]->(st:Step)
DETACH DELETE st
"""

_DROP_PARAMS_CYPHER = """
UNWIND $ids AS cid
MATCH (p:Parameter)-[:DEFINED_IN_TABLE]->(c:Chunk {chunk_id: cid})
DETACH DELETE p
"""

_CLEAR_EMBEDDING_CYPHER = """
UNWIND $ids AS cid
MATCH (c:Chunk {chunk_id: cid})
REMOVE c.embedding
"""

_CONTAINS_CYPHER = """
UNWIND $ids AS cid
MATCH (c:Chunk {chunk_id: cid})
MATCH (d:Document {spec_id: c.spec_id})
MERGE (d)-[:CONTAINS]->(c)
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Split tail-swallow chunks on a live KG")
    ap.add_argument("--apply", action="store_true", help="write to Neo4j (default: dry-run)")
    ap.add_argument("--limit", type=int, default=0, help="process at most N candidate chunks")
    ap.add_argument("--split-letter-clauses", action="store_true",
                    help="also split letter-suffixed clauses (7A, 5.8a) outside annexes")
    ap.add_argument("--drop-parameters", action="store_true",
                    help="delete Parameter nodes of repaired chunks (cannot be re-derived here)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    chunker = _load_chunker()
    parser = chunker.ThreeGPPParser()

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    auth = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))
    driver = GraphDatabase.driver(uri, auth=auth)

    touched_ids: list[str] = []      # existing chunks whose content shrank
    new_dicts: list[dict] = []       # chunks to create
    head_dicts: list[dict] = []      # rewritten heads
    n_seen = n_hit = n_hist = n_new_est = 0
    freed = 0

    try:
        for row in _iter_candidates(driver):
            n_seen += 1
            plan = plan_split(row["section_id"], row["section_title"], row["content"],
                              chunker, args.split_letter_clauses)
            if plan is None:
                continue
            head_content, tail, dropped_history = plan
            n_hit += 1
            n_hist += int(dropped_history)
            freed += len(row["content"]) - len(head_content) - sum(len(b.content) for b in tail)

            touched_ids.append(row["chunk_id"])
            n_new_est += len(tail)
            # Building the chunk dicts runs key-term extraction + complexity +
            # cross-reference parsing over every block — skip it unless we are
            # actually writing, so a dry-run over all ~1.9k candidates is quick.
            if args.apply:
                head_dicts += _to_chunk_dicts(row["spec_id"], row["section_id"],
                                              row["section_title"], head_content, chunker, parser)
                for b in tail:
                    new_dicts += _to_chunk_dicts(row["spec_id"], b.section_id, b.title,
                                                 b.content, chunker, parser)

            if args.verbose:
                print(f"  {row['chunk_id']}: {len(row['content']):>8} → head {len(head_content):>7}"
                      f" + {len(tail):>3} sections"
                      f"{'  [dropped change history]' if dropped_history else ''}")
            if args.limit and n_hit >= args.limit:
                break

        print(f"\ncandidates scanned : {n_seen}")
        print(f"chunks to repair   : {n_hit}  (change history dropped in {n_hist})")
        print(f"new chunks         : {n_new_est}")
        print(f"chars dropped      : {freed:,}")

        if not args.apply:
            print("\nDRY RUN — nothing written. Re-run with --apply.")
            return

        builder = KGBuilder(uri=uri, user=auth[0], password=auth[1])
        try:
            all_dicts = head_dicts + new_dicts
            all_ids = [d["chunk_id"] for d in all_dicts]

            with driver.session() as s:
                s.run(_PURGE_CYPHER, ids=touched_ids)
                if args.drop_parameters:
                    s.run(_DROP_PARAMS_CYPHER, ids=touched_ids)

            # Same methods the build pipeline uses — order mirrors load_json_dir.
            builder._create_chunks(all_dicts)
            with driver.session() as s:
                s.run(_CONTAINS_CYPHER, ids=all_ids)
                s.run(_CLEAR_EMBEDDING_CYPHER, ids=all_ids)
            builder._create_references_spec_edges(all_dicts)
            builder._create_references_chunk_edges(all_dicts)
            builder._create_terms(all_dicts)
            builder._create_subjects(all_dicts)
            builder._create_mentions_edges(all_dicts)
            builder._create_service_operations(all_dicts)
            builder._create_procedure_steps(all_dicts)
            builder._create_messages(all_dicts)
        finally:
            builder.close()

        print(f"\napplied: {len(head_dicts)} heads rewritten, {len(new_dicts)} chunks created.")
        print("Next: npm run rebuild-kg:embed-only   (embeds the cleared/new chunks)")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
