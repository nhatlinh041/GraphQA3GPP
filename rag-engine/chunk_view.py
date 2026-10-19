"""
POST /api/chunks/segments — the chat's source cards render tables.

`Chunk.content` stores tables flattened one cell per line. The source JSON
(processed_json_v6) still has them as headers + rows; review/tables.py locates
each flat run inside the content and returns text/table segments. Read-only:
nothing in Neo4j or in the SSE `sources` payload changes, the client asks for
segments of content it already has.
"""
from __future__ import annotations

import json
from functools import lru_cache

from fastapi import APIRouter
from pydantic import BaseModel

from review import tables as T

router = APIRouter(prefix="/api/chunks", tags=["chunks"])

MAX_ITEMS = 40
MAX_CONTENT = 200_000


@lru_cache(maxsize=64)
def _spec_tables(spec: str) -> dict[str, list[dict]]:
    """chunk_id -> tables, only for chunks that have any. One JSON parse per spec."""
    path = T.JSON_DIR / f"{spec}.json"
    if not path.is_file():
        return {}
    out: dict[str, list[dict]] = {}
    for ch in json.loads(path.read_text()).get("chunks") or []:
        ts = ch.get("tables") or []
        if ts:
            out[ch["chunk_id"]] = [
                {"headers": [str(h) for h in (t.get("headers") or [])],
                 "rows": [[str(c) for c in row] for row in (t.get("rows") or [])]}
                for t in ts
            ]
    return out


class Item(BaseModel):
    chunk_id: str
    content: str


class SegmentsIn(BaseModel):
    items: list[Item]


@router.post("/segments")
def chunk_segments(body: SegmentsIn) -> dict:
    out: dict[str, list[dict]] = {}
    for it in body.items[:MAX_ITEMS]:
        spec = "_".join(it.chunk_id.split("_")[:3])
        # spec ids are [a-z]{2}_\d+_[\w-]+ — refuse anything that could walk the filesystem
        if "/" in spec or ".." in spec or not spec:
            continue
        tables = _spec_tables(spec).get(it.chunk_id, [])
        out[it.chunk_id] = T.segments(it.content[:MAX_CONTENT], tables)
    return {"segments": out}
