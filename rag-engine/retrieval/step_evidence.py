"""
Step-scoped evidence — EXPERIMENT, default off (STEP_EVIDENCE=1 to enable).

The question this exists to answer: a procedure chunk averages 3,930 characters
across 8.8 steps, and 76% of that text IS the steps. If a question is answered by
one step, feeding the whole clause spends ~12x more context than the answer needs.
Chunk-level Recall cannot see that difference — BM25 retrieving the clause scores
exactly the same as a graph branch that could point at step 5 of 14 — which is why
`kg_procedure` measured KG 0.620 vs BM25 0.740 despite the graph holding strictly
more structure (v25, 2026-08-24).

So this does not try to improve recall. It replaces the delivered evidence with
just the steps that involve what the question asked about, and the experiment is
whether answer quality HOLDS while context shrinks. If quality drops, the whole
idea is wrong and the measurement says so; a context saving bought with worse
answers is not a saving.

Deliberately NOT a retrieval change: the same chunks are retrieved, only their
rendering changes. That keeps Recall@5 identical between on and off, so the two
runs differ in exactly one variable.
"""
import os
import re

STEP_EVIDENCE = os.getenv("STEP_EVIDENCE", "0") == "1"
# Steps either side of a matched one. A 3GPP step routinely says "the SMF
# acknowledges" with the subject only named in the step before, so a lone step is
# often unreadable; one neighbour each way is the cheapest fix that keeps it
# grounded. 0 disables the padding.
STEP_CONTEXT_NEIGHBOURS = int(os.getenv("STEP_CONTEXT_NEIGHBOURS", "1"))
# Below this many steps the clause is short enough that scoping saves nothing.
STEP_EVIDENCE_MIN_STEPS = int(os.getenv("STEP_EVIDENCE_MIN_STEPS", "4"))

_FETCH = """
UNWIND $ids AS cid
MATCH (c:Chunk {chunk_id: cid})-[:HAS_STEP]->(s:Step)
OPTIONAL MATCH (s)-[:INVOLVES]->(t:Term)
RETURN cid AS chunk_id, s.order AS ord, s.text AS text,
       collect(DISTINCT t.abbreviation) AS actors
"""


def _order_key(order: str):
    """'4a' sorts after '4' and before '5'; a non-numeric order sorts last."""
    m = re.match(r"(\d+)([a-z]?)", str(order or ""))
    if not m:
        return (10 ** 6, "")
    return (int(m.group(1)), m.group(2))


def attach_step_evidence(driver, chunks, question, resolved_terms):
    """Set c['step_evidence'] on chunks whose steps can be narrowed.

    A step is kept when it involves one of the question's resolved terms, or when
    its text contains one of them word-boundary — the INVOLVES edge is built from
    chunk-level MENTIONS and misses terms the step names but the chunk's key_terms
    dropped, so the textual check is a genuine second chance, not a duplicate.

    Chunks are left untouched (no key set) when nothing matches or when every step
    matches: replacing a clause with all of its own steps saves nothing and only
    strips the surrounding prose that says what the procedure IS.
    """
    if not STEP_EVIDENCE or not chunks:
        return 0
    wanted = {t.upper() for t in (resolved_terms or [])}
    if not wanted:
        return 0
    ids = [c.get("chunk_id") for c in chunks if c.get("chunk_id")]
    if not ids:
        return 0

    by_chunk = {}
    with driver.session() as s:
        for r in s.run(_FETCH, ids=ids):
            by_chunk.setdefault(r["chunk_id"], []).append(
                {"ord": r["ord"], "text": r["text"] or "",
                 "actors": {a.upper() for a in (r["actors"] or []) if a}})

    n_scoped = 0
    for c in chunks:
        steps = by_chunk.get(c.get("chunk_id")) or []
        if len(steps) < STEP_EVIDENCE_MIN_STEPS:
            continue
        steps.sort(key=lambda x: _order_key(x["ord"]))
        hits = set()
        for i, st in enumerate(steps):
            if st["actors"] & wanted:
                hits.add(i)
                continue
            up = st["text"].upper()
            if any(re.search(r"(?<![A-Z0-9])" + re.escape(w) + r"(?![A-Z0-9])", up)
                   for w in wanted):
                hits.add(i)
        if not hits or len(hits) == len(steps):
            continue
        keep = set()
        for i in hits:
            for j in range(i - STEP_CONTEXT_NEIGHBOURS, i + STEP_CONTEXT_NEIGHBOURS + 1):
                if 0 <= j < len(steps):
                    keep.add(j)
        if len(keep) == len(steps):
            continue
        body = "\n".join(f"{steps[i]['ord']}.\t{steps[i]['text']}" for i in sorted(keep))
        # Keep the clause title line so the excerpt still says what procedure it is.
        head = (c.get("section") or c.get("section_title") or "").strip()
        c["step_evidence"] = (f"{head}\n{body}" if head else body)
        c["step_evidence_stats"] = {
            "steps_total": len(steps), "steps_kept": len(keep),
            "chars_full": len(c.get("content") or ""), "chars_scoped": len(c["step_evidence"]),
        }
        n_scoped += 1
    return n_scoped


# ── Step-actor marking (separate from the scoping experiment above) ───────────
_STEP_ACTOR_CYPHER = """
UNWIND $ids AS cid
MATCH (c:Chunk {chunk_id: cid})-[:HAS_STEP]->(s:Step)-[:INVOLVES]->(t:Term)
WHERE t.abbreviation IN $terms
WITH cid, s, count(DISTINCT t.abbreviation) AS n_here
WHERE n_here >= $min_actors
RETURN cid AS chunk_id, count(DISTINCT s) AS n_steps
"""


def mark_step_actor_chunks(driver, chunks, resolved_terms, min_actors=2):
    """Mark chunks that have a Step where >=min_actors of the question's terms meet.

    Sets `via_step_actors` and `n_step_actor_hits`. fusion.py exempts marked chunks
    from the cross-encoder floor, so this must assert a STRUCTURAL fact and nothing
    weaker: the check is a graph query against Step/INVOLVES, never a guess from the
    generated Cypher, which may have used any pattern at all.

    Why the exemption is needed: "which procedures have the UDM and the NRF in the
    same step" is a question about the corpus, so the clause that answers it shares
    almost no wording with it and the cross-encoder — which scores exactly that
    overlap — rates it near zero. Measured on KGP-010: 27 candidates in, 0 through
    the floor, gold among the 27.

    min_actors=2 on purpose. At 1 this fires on nearly every procedure chunk (AMF
    alone appears in steps of 431 clauses) and the exemption stops meaning anything.
    """
    if not chunks or not resolved_terms:
        return 0
    terms = sorted({t.upper() for t in resolved_terms if t})
    if len(terms) < min_actors:
        return 0
    ids = [c.get("chunk_id") for c in chunks if c.get("chunk_id")]
    if not ids:
        return 0
    hits = {}
    with driver.session() as s:
        for r in s.run(_STEP_ACTOR_CYPHER, ids=ids, terms=terms, min_actors=min_actors):
            hits[r["chunk_id"]] = r["n_steps"]
    for c in chunks:
        n = hits.get(c.get("chunk_id"))
        if n:
            c["via_step_actors"] = True
            c["n_step_actor_hits"] = n
            # rerank_bias has read `has_step_all_actors` since it was written, with a
            # tuned weight of 0.7 — and NOTHING in the codebase ever set it. The boost
            # has never fired once, and the matching `op_prefix_match` flag is dead the
            # same way. Measured consequence on KGP-047 ("which procedures involve the
            # AMF, the NEF and the UPF at the same step"): four of the top five slots
            # went to architecture clauses TITLED "AMF", "NEF", "NEF functionality" —
            # none of which has a single step where all three meet — while four correct
            # clauses sat at ranks 6-10. Setting it here revives the existing feature
            # rather than adding a new one.
            c["has_step_all_actors"] = True
    return len(hits)
