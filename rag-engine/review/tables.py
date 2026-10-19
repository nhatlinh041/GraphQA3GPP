"""
Structured tables for the gold clauses shown to raters.

`Chunk.content` in Neo4j stores every table flattened, one cell per line, which
is unreadable for a 16-row IE table. The source JSON (processed_json_v6) still
carries each chunk's `tables` as headers + rows. We snapshot those next to the
gold text and, at display time, find the flat run of cells inside the clause
text and swap it for the table. Nothing in Neo4j or in the stored content is
changed.

Matching ignores ALL whitespace on both sides: the table extractor joins a
cell's paragraphs with nothing ("request(see NOTE)") while the flat text keeps
the newline, and non-breaking spaces differ too. Measured on the 813 gold
chunks of the review sample: 431/432 tables located; an unlocated table just
stays as flat text.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JSON_DIR = Path(os.getenv("REVIEW_JSON_DIR") or ROOT / "3GPP_JSON_DOC" / "processed_json_v6")
_WS = re.compile(r"\s+")


def load_tables(chunk_ids: list[str]) -> dict[str, list[dict]]:
    """chunk_id -> list of {headers, rows}; chunks with no table map to []."""
    by_spec: dict[str, set] = defaultdict(set)
    for c in chunk_ids:
        by_spec["_".join(c.split("_")[:3])].add(c)
    out: dict[str, list[dict]] = {}
    for spec, cids in by_spec.items():
        path = JSON_DIR / f"{spec}.json"
        if not path.is_file():
            continue
        for ch in json.loads(path.read_text()).get("chunks") or []:
            if ch.get("chunk_id") in cids:
                out[ch["chunk_id"]] = [
                    {"headers": [str(h) for h in (t.get("headers") or [])],
                     "rows": [[str(c) for c in row] for row in (t.get("rows") or [])]}
                    for t in (ch.get("tables") or [])
                ]
    return out


def _squeeze(s: str) -> tuple[str, list[int]]:
    keep, idx = [], []
    for k, ch in enumerate(s):
        if not ch.isspace():
            keep.append(ch)
            idx.append(k)
    return "".join(keep), idx


def _prefix_match(sq: str, pos: int, headers: list[str], rows: list[list[str]]) -> tuple[int, int, str]:
    """(start, n_complete_rows, matched_flat) for the longest header+rows prefix
    of the table present in `sq` from `pos`; (-1, 0, "") when not even the
    header and one full row are there."""
    head = "".join(_WS.sub("", c) for c in headers)
    first = "".join(_WS.sub("", c) for c in (rows[0] if rows else []))
    anchor = head + first
    if not rows or not anchor:
        return -1, 0, ""
    i = sq.find(anchor, pos)
    if i < 0:
        return -1, 0, ""
    flat, n = anchor, 1
    for row in rows[1:]:
        nxt = flat + "".join(_WS.sub("", c) for c in row)
        if not sq.startswith(nxt, i):
            break
        flat, n = nxt, n + 1
    return i, n, flat


def segments(content: str | None, tables: list[dict] | None) -> list[dict]:
    """Split the clause text into [{type: text, text}, {type: table, headers, rows}, …]."""
    content = content or ""
    if not tables:
        return [{"type": "text", "text": content}]
    sq, idx = _squeeze(content)
    out: list[dict] = []
    pos_sq, pos_orig = 0, 0
    for t in tables:
        headers, rows = t.get("headers") or [], t.get("rows") or []
        cells = list(headers) + [c for row in rows for c in row]
        flat = "".join(_WS.sub("", c) for c in cells)
        if not flat:
            continue
        i = sq.find(flat, pos_sq)
        n_rows, truncated = len(rows), False
        if i < 0:
            # The text may be an excerpt cut mid-table (the adaptive-hop path caps
            # chunks at 2000 chars): keep the header and every row that is complete.
            i, n_rows, flat = _prefix_match(sq, pos_sq, headers, rows)
            truncated = True
            if i < 0:
                continue
        start, end = idx[i], idx[i + len(flat) - 1] + 1
        if start > pos_orig:
            out.append({"type": "text", "text": content[pos_orig:start]})
        seg = {"type": "table", "headers": headers, "rows": rows[:n_rows]}
        if truncated:
            seg["truncated"] = True
        out.append(seg)
        pos_sq, pos_orig = i + len(flat), end
    if pos_orig < len(content):
        out.append({"type": "text", "text": content[pos_orig:]})
    return out
