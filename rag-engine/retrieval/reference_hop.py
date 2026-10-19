"""
Reference-chain hop expansion — follow REFERENCES_CHUNK 1-2 hops from chunks
retrieval already found, to recover cross-spec cited chunks that vector search
and the Term/Step-anchored Cypher patterns (A1/A2/B/F/G/M/P) cannot reach.

Why this exists: de_cuong/PhuLuc_B_5ca_vector_vs_fixed.md measured 5 questions
where the answer chunk B sits behind a REFERENCES_CHUNK edge from a chunk A that
retrieval DOES find (cosine(A,B) as low as 0.78, B ranked >15,000/182,961 when
querying by vector(A) alone) — B never enters the candidate pool in either
`vector_only` or `fixed` mode, because every existing Cypher pattern re-anchors
on the QUESTION's terms/steps, not on chunks retrieval already surfaced. None of
them implement "chunk A cites chunk B, go get B."

Why this is a NEW module rather than resurrecting `multihop_search.py`: that
module's patterns seed from a raw `Term.abbreviation` (`MATCH (t:Term
{abbreviation: $seed}) ... start.spec_id IN t.source_specs`) — the same
"anchor on a Term, fan out to every chunk in its source_specs" shape already
diagnosed elsewhere in this codebase (Pattern F/G's constant-score flood,
BAO_CAO_5_MODE.md) as producing hundreds of candidates with an arbitrary LIMIT
cut. Seeding from a handful of chunks retrieval has ALREADY ranked relevant
(post vector+graph, pre-final-rerank) keeps the UNWIND small and targeted.
"""
from __future__ import annotations

from typing import Optional

# Per-SEED row cap (not a global cap — see _hop_cypher). Measured against the
# live KG: a single chunk's REFERENCES_CHUNK out-degree for this 1-2 hop
# cross-spec pattern can reach ~929 rows (p99=101 across chunks with any
# outgoing edge). A flat global LIMIT applied over the UNWIND across all seeds
# combined would let one such high-fan-out seed consume the entire budget
# and starve every other seed to zero rows BEFORE the round-robin below ever
# runs — reproducing, one Cypher stage earlier, the exact "one seed crowds out
# another" failure the round-robin exists to prevent. Applying the LIMIT
# INSIDE a per-seed CALL subquery instead guarantees every seed contributes at
# most this many rows regardless of its own fan-out.
_PER_SEED_ROW_CAP = 50


def _hop_cypher(max_hops: int) -> str:
    # max_hops is an internal constant (never user input) — inlined because
    # Cypher variable-length bounds (`*1..N`) aren't parameterizable.
    return f"""UNWIND $seeds AS sid
CALL (sid) {{
  MATCH p = (seed:Chunk {{chunk_id: sid}})-[:REFERENCES_CHUNK*1..{max_hops}]->(target:Chunk)
  WHERE target.spec_id <> seed.spec_id
    AND size(target.content) >= 100
  RETURN target.chunk_id AS chunk_id, target.content AS content,
         target.spec_id AS spec_id, target.section_title AS section,
         target.chunk_type AS chunk_type, sid AS via_seed, length(p) AS hops
  ORDER BY hops ASC
  LIMIT $per_seed_cap
}}
RETURN chunk_id, content, spec_id, section, chunk_type, via_seed, hops"""


def expand_reference_chain(
    driver,
    seed_chunk_ids: list[str],
    max_hops: int = 2,
    max_targets: int = 40,
) -> list[dict]:
    """Follow REFERENCES_CHUNK 1..max_hops from each seed chunk_id, return
    cross-spec target chunks as candidate dicts (same shape as other retrieval
    branches: chunk_id/content/spec_id/section/score). Deduped by chunk_id
    (keeps the shortest-hop occurrence when a target is reachable from
    multiple seeds or hop counts). Never raises — returns [] on any KG error
    so a broken query degrades to "no hop expansion", not a failed request.

    Marked `score=0.0, via_reference_chain=True`. The floor bypass (fusion.py's
    RERANK_HOP_FLOOR_BYPASS) and the rerank_bias boost (W_REFERENCE_CHAIN_BOOST)
    are keyed off `via_reference_chain`, NOT `score` — these chunks are found
    via an EXPLICIT authored citation (REFERENCES_CHUNK, confidence>=0.7 at
    KG-build time), not semantic similarity to the question, so the
    cross-encoder logit alone must not be the sole gate deciding whether they
    reach the answer context. `score` is deliberately kept LOW (not 1.0):
    fusion.py's blend also reads `score` as "upstream confidence" for the
    UNRELATED `is_canonical` mechanism (Pattern A1/B exact section-title
    match), which adds its own +0.3 upstream-weight and +0.15 canonical-bonus
    to `final_score`. Setting score=1.0 here would silently trigger THAT bonus
    too, stacking ~0.45 extra points on top of the dedicated floor-bypass and
    boost — every hop chunk would cluster near ~0.7 regardless of actual
    relevance (measured live: this compressed real differences between a
    genuinely-relevant hop chunk and a tangential one enough to risk evicting
    a genuinely relevant non-hop chunk from top_k). score=0.0 keeps the two
    mechanisms decoupled — via_reference_chain earns its bypass + boost, but a
    hop chunk still has to earn ranking headroom via a positive cross-encoder
    logit like everything else.
    """
    seeds = [s for s in dict.fromkeys(seed_chunk_ids) if s]
    if not seeds:
        return []
    try:
        with driver.session(default_access_mode="READ") as session:
            records = session.run(
                _hop_cypher(max_hops), seeds=seeds, per_seed_cap=_PER_SEED_ROW_CAP,
            )
            rows = [dict(r) for r in records]
    except Exception:
        return []

    best: dict[str, dict] = {}
    for r in rows:
        cid = r.get("chunk_id")
        if not cid:
            continue
        if cid not in best or (r.get("hops") or 99) < (best[cid].get("hops") or 99):
            best[cid] = r

    # Interleave round-robin across seeds (each seed's own targets sorted by hop
    # count) before applying max_targets — a flat sort-then-slice would let
    # whichever seed happens to have the most outbound citations (or whichever
    # Neo4j returns first) crowd out a DIFFERENT seed's single important
    # citation. This is the exact "arbitrary order + LIMIT cut drops the gold"
    # failure this codebase already diagnosed for Pattern F/G (see
    # orchestrator.py TOP_K_GRAPH_WIDE) — round-robin guarantees every seed
    # gets a fair share of the cap instead of the first-seen seed dominating.
    from itertools import zip_longest
    by_seed: dict[str, list[dict]] = {}
    for r in best.values():
        by_seed.setdefault(r.get("via_seed") or "?", []).append(r)
    for lst in by_seed.values():
        lst.sort(key=lambda r: r.get("hops") or 99)
    ranked = [
        r for group in zip_longest(*by_seed.values()) for r in group if r is not None
    ][:max_targets]
    return [{
        "chunk_id": r["chunk_id"],
        "content": r.get("content") or "",
        "spec_id": r.get("spec_id") or "?",
        "section": r.get("section") or "?",
        "chunk_type": r.get("chunk_type"),
        "score": 0.0,
        "via_reference_chain": True,
        "hop_via_seed": r.get("via_seed"),
        "hops": r.get("hops"),
    } for r in ranked]
