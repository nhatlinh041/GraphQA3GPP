"""
Anchor retrieval — goal.md G4. When a question is about a named procedure
(registration, handover, authentication, …), pull the procedure's ROOT clause
straight out of the KG by section_title, in parallel with vector + graph search.

Why this exists: the five-round AMF↔UDM case study found that TS 23.502
§4.2.2.2.2 "General Registration" — the clause that defines the standard
registration flow (Nudm_UECM_Registration → SDM_Get → SDM_Subscribe) — never
reached the answer context through any run. It is a `definition`-typed chunk with
NO Step, so it loses the FG structural score to procedure-typed variant chunks,
and the cross-encoder doesn't recognise it as the canonical flow. Neither vector
nor graph traversal surfaces it reliably. Anchor retrieval fetches it directly by
matching the question's procedure name against section_title, restricted to
normative TS specs (not TRs), and marks it `parent_of_anchor=True` so the Part E
rerank bias boosts it.

Deliberately conservative: fires only for a known procedure keyword, matches
section_title exactly or as "<proc>" / "general <proc>" / "<proc> procedure",
caps the number of anchors, and prefers architecture/core specs. A miss returns
[] and the pipeline is unchanged.
"""
from __future__ import annotations

import os
from typing import Optional

# Procedure keyword → the section-title forms its root clause tends to use. The
# first spec list is the preferred normative home for that procedure's root flow.
# Kept small and high-precision; extend as new procedures need anchoring.
_PROCEDURE_ANCHORS: dict[str, dict] = {
    "registration": {
        "titles": ["general registration", "registration", "registration procedure"],
        "specs": ["ts_23_502", "ts_24_501"],
    },
    "deregistration": {
        "titles": ["deregistration", "deregistration procedure", "general deregistration"],
        "specs": ["ts_23_502", "ts_24_501"],
    },
    "handover": {
        "titles": ["handover", "handover procedure", "general handover"],
        "specs": ["ts_23_502", "ts_33_501"],
    },
    "authentication": {
        "titles": ["authentication", "authentication procedure",
                   "primary authentication and key agreement procedure"],
        "specs": ["ts_33_501", "ts_23_502"],
    },
    "service request": {
        "titles": ["ue triggered service request", "service request",
                   "network triggered service request"],
        "specs": ["ts_23_502"],
    },
    "pdu session establishment": {
        "titles": ["ue requested pdu session establishment",
                   "pdu session establishment"],
        "specs": ["ts_23_502"],
    },
    "pdu session modification": {
        "titles": ["ue or network requested pdu session modification",
                   "pdu session modification"],
        "specs": ["ts_23_502"],
    },
}

# Longest keys first so "pdu session establishment" wins over a bare "session".
_ANCHOR_KEYS = sorted(_PROCEDURE_ANCHORS, key=len, reverse=True)

MAX_ANCHORS = 3

# A hard anchor is front-pulled ahead of the cross-encoder verdict, so an
# oversized chunk costs a context slot AND mis-attributes its excerpt: the
# chunker's tail-swallow bug (a document's last numbered clause absorbs every
# Annex + the change history — see document_processing section_pattern) produces
# chunks up to 1.5 MB whose _fit_chunk_content excerpt is real spec text carrying
# a completely unrelated section citation. Measured live: ts_33_501_16.6.3
# (262 KB, titled "Subscription/unsubscription of NSACF notification") anchored a
# 5G AKA question and fed the answer the right steps under the wrong §.
# Oversized anchors are demoted to soft — still in the candidate pool, but they
# must earn their place from the reranker. 40 KB is far above any genuine clause
# (median chunk is ~1 KB; only 1.7k of 183k chunks exceed 20 KB).
MAX_HARD_ANCHOR_CHARS = int(os.getenv("ANCHOR_MAX_HARD_CHARS", "40000"))


def detect_procedure(question: str) -> Optional[str]:
    """Return the procedure keyword the question is about, or None."""
    q = (question or "").lower()
    for key in _ANCHOR_KEYS:
        if key in q:
            return key
    return None


def _anchor_cypher(titles: list[str], specs: list[str]) -> str:
    title_list = "[" + ", ".join(f"'{t}'" for t in titles) + "]"
    spec_list = "[" + ", ".join(f"'{s}'" for s in specs) + "]"
    # Exact (lower-cased) section_title match, restricted to the normative specs.
    # Ordered so the earliest-listed title form and spec win, with a stable
    # chunk_id tie-break (G10). Returns the same column contract as the graph
    # patterns + the parent_of_anchor flag the rerank bias reads.
    return f"""MATCH (c:Chunk)
WHERE toLower(c.section_title) IN {title_list}
  AND c.spec_id IN {spec_list}
RETURN c.chunk_id AS chunk_id, c.content AS content, c.spec_id AS spec_id,
       c.section_title AS section, c.chunk_type AS chunk_type,
       1.0 AS score, true AS parent_of_anchor
ORDER BY c.spec_id, c.chunk_id
LIMIT {MAX_ANCHORS}"""


import re

# Named-section phrases the question quotes verbatim. wh_kg_v3 questions often
# name the exact procedure section: "in the 'Setting up an AF session with
# required QoS' procedure of TS 23.502…". That quoted phrase IS a section_title
# (gold ts_23_502_4.15.6.6), but neither vector nor the FG graph flood surfaces
# it — the FG branch pulls every AF+NEF chunk and the gold sinks. Matching the
# quoted phrase against section_title lands the gold directly. Captures single-
# and double-quoted spans of 3+ words (short quotes are too ambiguous).
_QUOTED_PHRASE_RE = re.compile(r"['\"‘’“”]([A-Za-z][A-Za-z0-9 /\-]{12,80})['\"‘’“”]")


def _quoted_section_phrases(question: str) -> list[str]:
    """Verbatim section-name phrases the question quotes (3+ words)."""
    out = []
    for m in _QUOTED_PHRASE_RE.finditer(question or ""):
        p = m.group(1).strip()
        if len(p.split()) >= 3:
            out.append(p)
    return out


def _phrase_anchor_cypher(phrase: str) -> str:
    from_esc = phrase.replace("\\", "\\\\").replace("'", "\\'")
    # section_title CONTAINS the quoted phrase (case-insensitive). Prefer the
    # shortest matching title (the exact section, not a longer variant that
    # merely embeds the phrase) and a stable chunk_id tie-break.
    return f"""MATCH (c:Chunk)
WHERE toLower(c.section_title) CONTAINS toLower('{from_esc}')
RETURN c.chunk_id AS chunk_id, c.content AS content, c.spec_id AS spec_id,
       c.section_title AS section, c.chunk_type AS chunk_type,
       1.0 AS score, true AS parent_of_anchor, size(c.section_title) AS tlen
ORDER BY tlen ASC, c.chunk_id ASC
LIMIT {MAX_ANCHORS}"""


# SBI service-operation names the question quotes (Nxxx_Yyy_Zzz). A Pattern-G
# question ("...through Nnwdaf_AnalyticsInfo_Request", "how do Nhss_...Update and
# Nudm_...Update correspond?") names the operations verbatim, and the gold is the
# chunk that DESCRIBES those operations — often the ONLY chunk describing all of
# them together. Structural score can't surface it (a popular operation appears in
# dozens of chunks), but an exact DESCRIBES_OPERATION match on the named set can.
_OP_NAME_RE = re.compile(r"\bN[a-z0-9]{2,6}_[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+\b")


def _operation_names(question: str) -> list[str]:
    seen, out = set(), []
    for m in _OP_NAME_RE.finditer(question or ""):
        n = m.group(0)
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out[:4]


def _operation_anchor_cypher(op_names: list[str]) -> str:
    # Per-operation coverage (2026-08-08 rework). The old query required a single
    # chunk to describe ALL named ops (n_matched >= 2) — but on wh_kg_v3 most
    # two-op questions have the two ops defined in two DIFFERENT chunks (each
    # describing one op), so the all-match filter dropped the gold and the
    # single-op fallback returned 3 chunks in arbitrary chunk_id order out of
    # dozens. Instead: for EACH named op take the top-2 chunks, preferring the
    # op's own defining clause (section_title contains the op name — 3GPP stage-2/3
    # specs title the operation's clause with its name). `title_score` and
    # `n_matched` come back so the caller can grade trust: defining clauses and
    # all-op matches are hard anchors; the rest join the pool as soft anchors.
    lst = "[" + ", ".join("'" + n.replace("'", "\\'") + "'" for n in op_names) + "]"
    # Stage-3 (ts_29.xxx) specs title an operation's subclause with just the final
    # segment ("ConfigCreate", "Subscribe") — score 1. That's safe here because the
    # chunk already DESCRIBES the exact op, so a generic verb in the title still
    # identifies the op's own subclause, not an unrelated section.
    return f"""UNWIND {lst} AS opname
MATCH (c:Chunk)-[:DESCRIBES_OPERATION]->(:ServiceOperation {{name: opname}})
WITH opname, c,
     (CASE WHEN toLower(c.section_title) CONTAINS toLower(opname) THEN 2
           WHEN size(last(split(opname, '_'))) >= 4
                AND toLower(c.section_title) CONTAINS toLower(last(split(opname, '_'))) THEN 1
           ELSE 0 END) AS title_score
OPTIONAL MATCH (c)-[:DESCRIBES_OPERATION]->(so2:ServiceOperation)
WHERE so2.name IN {lst}
WITH opname, c, title_score, count(DISTINCT so2.name) AS n_matched
ORDER BY title_score DESC, n_matched DESC, size(c.content) ASC
WITH opname, collect({{cid: c.chunk_id, content: c.content, spec_id: c.spec_id,
                      section: c.section_title, chunk_type: c.chunk_type,
                      title_score: title_score, n_matched: n_matched}})[..2] AS best
UNWIND best AS b
RETURN b.cid AS chunk_id, b.content AS content, b.spec_id AS spec_id,
       b.section AS section, b.chunk_type AS chunk_type,
       1.0 AS score, true AS parent_of_anchor,
       b.title_score AS title_score, b.n_matched AS n_matched"""


def _all_match_op_cypher(op_names: list[str]) -> str:
    # The pre-2026-08-08 semantics, kept as its own hard-anchor query: a single
    # chunk describing ALL named ops is usually the interaction clause a
    # multi-op (Pattern G) question asks about — measured hitGold 17/26 on
    # wh_kg_v3 group G. The per-op query above serves the opposite case (each
    # op defined in its own clause, Pattern O); the two are complementary.
    #
    # Tie-break matters more than the match here: every row scores n_matched=2,
    # so the old `chunk_id ASC` fallback was pure alphabet. Measured live on the
    # AUSF↔UDM 5G AKA question (2026-08-12): 12 chunks tied, and alphabet handed
    # the 3 hard-anchor slots to ts_23_316_7.2.1.3 (FN-RG registration), the
    # 262 KB tail-swallow chunk ts_33_501_16.6.3, and ts_33_501_6.1.2 — while
    # ts_33_501_6.1.3.2.0 "5G AKA" and 6.1.3.3.2 (both gold) sat 5th/6th and were
    # cut by LIMIT 3. Ordered buckets, worst-class last:
    #   1. oversized (MAX_HARD_ANCHOR_CHARS) — contains both ops because it
    #      contains half the spec, not because it is about them;
    #   2. TR (spec series >= 700) — a study/solution document, never the
    #      normative flow (same signal as rerank_bias._is_tr);
    #   3. then smallest first: a clause describing both ops in 1.5 KB is about
    #      that interaction; a 42 KB one merely mentions it.
    # This does NOT by itself put a method-specific gold ("5G AKA") first — that
    # needs a method anchor keyed on section_title — but it does stop the two
    # pathological classes from taking the lead.
    lst = "[" + ", ".join("'" + n.replace("'", "\\'") + "'" for n in op_names) + "]"
    return f"""MATCH (c:Chunk)-[:DESCRIBES_OPERATION]->(so:ServiceOperation)
WHERE so.name IN {lst}
WITH c, count(DISTINCT so.name) AS n_matched
WHERE n_matched >= 2
WITH c, n_matched,
     CASE WHEN size(c.content) > {MAX_HARD_ANCHOR_CHARS} THEN 1 ELSE 0 END AS oversized,
     CASE WHEN toInteger(left(split(c.spec_id, '_')[2], 3)) >= 700 THEN 1 ELSE 0 END AS is_tr
RETURN c.chunk_id AS chunk_id, c.content AS content, c.spec_id AS spec_id,
       c.section_title AS section, c.chunk_type AS chunk_type,
       1.0 AS score, true AS parent_of_anchor
ORDER BY n_matched DESC, oversized ASC, is_tr ASC, size(c.content) ASC, c.chunk_id ASC
LIMIT {MAX_ANCHORS}"""


# Service-level SBI names ("Nnwdaf_MLModelMonitor", "Npcf_UEPolicyControl") — the
# question names a service but not a full <svc>_<Operation> name. Match every
# operation of that service and prefer the chunk describing most of them (the
# service's overview/defining clause). Low trust: soft anchor only.
_SVC_NAME_RE = re.compile(r"\bN[a-z0-9]{2,6}_[A-Za-z0-9]+\b")


def _service_names(question: str, ops: list[str]) -> list[str]:
    seen, out = set(), []
    for m in _SVC_NAME_RE.finditer(question or ""):
        n = m.group(0)
        # Skip full op names and prefixes of an already-captured op.
        if n in seen or n in ops or any(o.startswith(n + "_") for o in ops):
            continue
        seen.add(n)
        out.append(n)
    return out[:2]


def _service_anchor_cypher(svc: str) -> str:
    esc = svc.replace("'", "\\'")
    return f"""MATCH (c:Chunk)-[:DESCRIBES_OPERATION]->(so:ServiceOperation)
WHERE so.name STARTS WITH '{esc}_'
WITH c, count(DISTINCT so.name) AS n_ops,
     (CASE WHEN toLower(c.section_title) CONTAINS toLower('{esc}') THEN 2 ELSE 0 END) AS title_score
ORDER BY title_score DESC, n_ops DESC, size(c.content) ASC
RETURN c.chunk_id AS chunk_id, c.content AS content, c.spec_id AS spec_id,
       c.section_title AS section, c.chunk_type AS chunk_type,
       1.0 AS score, true AS parent_of_anchor, 0 AS title_score, 1 AS n_matched
LIMIT 3"""


# Explicit clause citations — "TS 23.700-89 clause 6.7.2" / "clause 6.8.3 of
# TS 23.700-86". chunk_id is spec_id + '_' + clause number, so the citation
# resolves to a direct chunk lookup: the highest-precision anchor possible
# (measured 13/13 gold on wh_kg_v3 group F). Users of 3GPP documents cite
# clauses verbatim all the time; no retrieval pattern exploited that before.
_CLAUSE_REF_RE = re.compile(
    r"\bTS\s*(\d{2})\.(\d{3}(?:-\d+)?)\s*,?\s*(?:clause|section|§)\s*(\d+(?:\.\d+)*)", re.I)
_CLAUSE_REF_RE2 = re.compile(
    r"\b(?:clause|section|§)\s*(\d+(?:\.\d+)*)\s*(?:of|in)\s*TS\s*(\d{2})\.(\d{3}(?:-\d+)?)", re.I)


def _clause_ref_chunk_ids(question: str) -> list[str]:
    ids = []
    for m in _CLAUSE_REF_RE.finditer(question or ""):
        ids.append(f"ts_{m.group(1)}_{m.group(2)}_{m.group(3)}")
    for m in _CLAUSE_REF_RE2.finditer(question or ""):
        ids.append(f"ts_{m.group(2)}_{m.group(3)}_{m.group(1)}")
    return list(dict.fromkeys(ids))[:3]


# ── Value-anchor (Layer C, 2026-08-09) ──────────────────────────────────────
# Standardized-value concepts (SST, 5QI, PQI, cause registries, OpenAPI enums)
# live in the KG as (Concept)-[:HAS_VALUE]->(StandardizedValue)-[:DEFINED_IN_TABLE]->
# (Chunk). A question naming such a concept ("which SST should we configure",
# "what does cause value 8 mean") anchors to the concept's defining table chunk —
# the exact failure mode this fixes: the SST question retrieved zero chunks
# containing the standardized values because no query path exploited value
# tables (measured live 2026-08-09; gold ts_23_501_5.15.2.2 unreachable).
# The Concept registry is small (~hundreds) — cache it with a TTL and match
# word-boundary in Python (server-side CONTAINS would hit "cause" in "because").
import time

_VALUE_REGISTRY: dict = {"ts": 0.0, "names": []}
_VALUE_REGISTRY_TTL_S = 600.0

# Concept names that read as ordinary English prose words must never drive the
# anchor: measured live 2026-08-09, the Concept 'Cause' matched the word
# "cause" in unrelated questions and its HARD anchor evicted the real gold from
# the top-5 ('Operation' did the same on three questions). The registry data
# stays in the KG — this filter only governs question matching.
_COMMON_WORD_STOP = {
    "cause", "operation", "status", "type", "event", "service", "name",
    "action", "result", "state", "mode", "direction", "priority", "category",
    "format", "method", "level", "class", "unit", "period", "condition",
    "indication", "notification", "report", "request", "response", "trigger",
    "reason", "source", "target", "value", "message", "parameter", "identity",
}
_TITLECASE_WORD_RE = re.compile(r"^[A-Z][a-z]+$")


def _anchorable_concept(name: str) -> bool:
    """Distinctive enough to match against free prose: not a bare common word
    (any casing), not a single TitleCase word ('Operation', 'Enumerated')."""
    if name.lower() in _COMMON_WORD_STOP:
        return False
    if _TITLECASE_WORD_RE.match(name):
        return False
    return True


def _value_concept_names(driver) -> list[str]:
    now = time.time()
    if now - _VALUE_REGISTRY["ts"] > _VALUE_REGISTRY_TTL_S:
        try:
            with driver.session() as s:
                _VALUE_REGISTRY["names"] = [
                    r["n"] for r in s.run(
                        "MATCH (co:Concept) WHERE size(co.name) >= 3 "
                        "RETURN co.name AS n")
                    if _anchorable_concept(r["n"])
                ]
            _VALUE_REGISTRY["ts"] = now
        except Exception:
            _VALUE_REGISTRY["ts"] = now  # do not hammer a down/empty DB
    return _VALUE_REGISTRY["names"]


_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _concept_patterns(name: str) -> list[str]:
    """Regex alternatives a question might use for this Concept name.

    Concept names come from table headers and are written the way a schema
    writes them -- PositioningMethod, EventType, NgapIeType -- while a person
    asks about "positioning methods" and "event types". Verbatim matching alone
    therefore never fires for any compound concept: measured on kg_value,
    23 of 58 questions detected NO concept and the graph branch scored exactly
    0.000 on all of them, while the concepts whose names people do type
    verbatim (5QI, PQI, SST, CQI) scored 0.688-1.000.

    Splitting on the camel-case boundary and tolerating a plural covers that.
    A split form is only offered when it yields >= 2 words and >= 8 characters:
    single split words land back in _COMMON_WORD_STOP territory ("Type",
    "Event"), but the two-word phrase is specific enough to anchor on.
    """
    pats = [re.escape(name)]
    words = _CAMEL_SPLIT_RE.split(name)
    # ALLCAPS compounds have no camel boundary to split on ("IETYPE"), so also
    # cut a trailing schema suffix off them.
    if len(words) == 1:
        m = re.fullmatch(r"([A-Z]{2,})(TYPE|CODE|NAME|ID)", name)
        if m:
            words = [m.group(1), m.group(2)]
    if len(words) >= 2 and len(name) >= 6:
        pats.append(r"\s+".join(re.escape(w) for w in words))
        # Schema names abbreviate what prose spells out: IETYPE / NgapIeType are
        # both asked about as "information element type". Only IE is expanded --
        # it is the one abbreviation that appears inside these concept names and
        # the phrase is specific enough not to over-fire (checked against
        # wh_kg_v5_600: see the false-positive count in the commit note).
        expanded = ["information element" if w.upper() == "IE" else w for w in words]
        if expanded != words:
            pats.append(r"\s+".join(re.escape(w) for w in expanded))
    # trailing plural: "positioning methods", "event types"
    return [p + r"s?" for p in pats]


def detect_value_concepts(driver, question: str) -> list[str]:
    """Concept names the question mentions, verbatim or in the spaced-out form a
    person would write (word-boundary, case-insensitive, plural tolerated).
    Empty list when the value layer is absent from the KG."""
    q = question or ""
    out = []
    for name in _value_concept_names(driver):
        for pat in _concept_patterns(name):
            if re.search(r"(?<![A-Za-z0-9])" + pat + r"(?![A-Za-z0-9])", q, re.IGNORECASE):
                out.append(name)
                break
    # Longest first — "EPS QCI" should outrank a bare "QCI" when both match.
    out.sort(key=len, reverse=True)
    return out[:3]


# Per concept: top-2 provenance chunks, preferring chunks whose value names the
# question also mentions (val_hits), then the value-richest table (the defining
# clause, e.g. ts_23_501_5.15.2.2 carries all 6 SST rows).
_VALUE_ANCHOR_CYPHER = """
UNWIND $concepts AS cname
MATCH (co:Concept {name: cname})-[:HAS_VALUE]->(v:StandardizedValue)
      -[:DEFINED_IN_TABLE]->(c:Chunk)
WITH cname, c,
     count(DISTINCT v) AS n_vals,
     count(DISTINCT CASE WHEN size(v.name) >= 4
                          AND toLower($q) CONTAINS toLower(v.name)
                    THEN v END) AS val_hits
ORDER BY val_hits DESC, n_vals DESC, c.chunk_id ASC
WITH cname, collect({cid: c.chunk_id, content: c.content, spec_id: c.spec_id,
                     section: c.section_title, chunk_type: c.chunk_type})[..2] AS best
UNWIND best AS b
RETURN b.cid AS chunk_id, b.content AS content, b.spec_id AS spec_id,
       b.section AS section, b.chunk_type AS chunk_type,
       1.0 AS score, true AS parent_of_anchor
"""


def _clause_anchor_cypher(chunk_ids: list[str]) -> str:
    lst = "[" + ", ".join(f"'{i}'" for i in chunk_ids) + "]"
    # Exact id first; fall back to direct children ("6.7.2.1") when the cited
    # clause itself isn't a chunk (some clauses only exist as their subclauses).
    return f"""MATCH (c:Chunk)
WHERE c.chunk_id IN {lst}
   OR any(id IN {lst} WHERE c.chunk_id STARTS WITH id + '.')
WITH c, (CASE WHEN c.chunk_id IN {lst} THEN 0 ELSE 1 END) AS depth
ORDER BY depth ASC, c.chunk_id ASC
RETURN c.chunk_id AS chunk_id, c.content AS content, c.spec_id AS spec_id,
       c.section_title AS section, c.chunk_type AS chunk_type,
       1.0 AS score, true AS parent_of_anchor
LIMIT {MAX_ANCHORS}"""


# Intents for which the procedure-ROOT anchor is allowed to fire. Kept aligned
# with the orchestrator's _PROC_BIAS_INTENTS: "what is the Registration
# Management state machine" contains "registration" but wants a definition, not
# the procedure root clause. The verbatim anchors (quoted phrase, SBI operation,
# clause citation) are intent-independent — they match text the question quotes
# literally, which is a precision signal on its own.
_PROC_ANCHOR_INTENTS = {"relationship", "procedure", "how_does"}


def fetch_anchor_chunks(driver, question: str, intent: str | None = None) -> list[dict]:
    """Fetch anchor chunk(s) for the question: (1) the procedure ROOT clause when
    a known procedure keyword is present (gated to procedure-ish intents when an
    intent is given), (2) any section whose title the question quotes verbatim,
    (3) chunks describing the SBI operations/services the question names, (4) the
    chunk a "TS xx.yyy clause z" citation addresses directly, (5) the defining
    table chunk of a standardized-value Concept the question names (SST/5QI/cause
    registries — Layer C). Returns chunk dicts
    with parent_of_anchor=True, deduped by chunk_id, or [] when nothing matches.
    Rows with soft_anchor=True are pool-coverage candidates only — the caller
    must NOT force them to the front of the context (they are per-op/service
    matches that may sit beside, not on, the answer). Never raises."""
    # (cypher, params, soft) triples — soft rows join the candidate pool but are
    # not trusted above the cross-encoder.
    cyphers: list[tuple[str, dict, bool]] = []
    proc = detect_procedure(question)
    if proc and (intent is None or intent in _PROC_ANCHOR_INTENTS):
        spec = _PROCEDURE_ANCHORS[proc]
        cyphers.append((_anchor_cypher(spec["titles"], spec["specs"]), {}, False))
    for phrase in _quoted_section_phrases(question):
        cyphers.append((_phrase_anchor_cypher(phrase), {}, False))
    clause_ids = _clause_ref_chunk_ids(question)
    if clause_ids:
        cyphers.append((_clause_anchor_cypher(clause_ids), {}, False))
    value_concepts = detect_value_concepts(driver, question)
    if value_concepts:
        cyphers.append((_VALUE_ANCHOR_CYPHER,
                        {"concepts": value_concepts, "q": question or ""}, False))
    ops = _operation_names(question)
    if ops:
        # All-match first: on a multi-op interaction question (Pattern G) the
        # chunk describing ALL named ops IS the asked-about clause, so it must
        # lead. A per-op defining clause promoted to hard lands at anchor rank
        # 4-5 — still inside the Recall@5 window for a Pattern-O question.
        if len(ops) >= 2:
            cyphers.append((_all_match_op_cypher(ops), {}, False))
        cyphers.append((_operation_anchor_cypher(ops), {}, True))
    for svc in _service_names(question, ops):
        cyphers.append((_service_anchor_cypher(svc), {}, True))
    if not cyphers:
        return []
    out, seen = [], set()
    n_ops = len(ops)
    try:
        with driver.session() as s:
            for cy, params, soft in cyphers:
                for r in s.run(cy, **params):
                    cid = str(r.get("chunk_id"))
                    if cid in seen:
                        continue
                    seen.add(cid)
                    # An op-anchor row is promoted to hard when it is the op's
                    # own defining clause (title names the op) or one chunk
                    # covers every named op (the old all-match semantics).
                    is_soft = soft
                    if soft and r.get("title_score") is not None:
                        title_hit = (r.get("title_score") or 0) >= 1
                        all_match = n_ops >= 2 and (r.get("n_matched") or 0) >= n_ops
                        if title_hit or all_match:
                            is_soft = False
                    content = r.get("content") or ""
                    # Size guard (see MAX_HARD_ANCHOR_CHARS): a tail-swallow chunk
                    # matches anchors for the wrong reason — it contains half the
                    # spec — so never let one lead the context.
                    if len(content) > MAX_HARD_ANCHOR_CHARS:
                        is_soft = True
                    out.append({
                        "chunk_id": cid,
                        "content": content,
                        "spec_id": r.get("spec_id") or "?",
                        "section": r.get("section") or "?",
                        "chunk_type": r.get("chunk_type"),
                        "score": 1.0,
                        "parent_of_anchor": not is_soft,
                        "soft_anchor": is_soft,
                    })
    except Exception:
        pass
    # Cap the hard anchors at 5 — Step 3a0 front-pulls every hard anchor, and a
    # lead longer than the Recall@5 window can only displace reranked hits.
    n_hard = 0
    capped = []
    for a in out:
        if not a.get("soft_anchor"):
            n_hard += 1
            if n_hard > 5:
                a = {**a, "parent_of_anchor": False, "soft_anchor": True}
        capped.append(a)
    return capped


# ---------------------------------------------------------------------------
# Pattern H — clause-tree aggregation
# ---------------------------------------------------------------------------
# A requirement whose answer is split across sibling clauses has no entity for
# the graph to anchor on: stage-1 service specs contribute 0.2% of Step nodes and
# none of the Message / ServiceOperation / StandardizedValue nodes. The only
# aggregating edge available is PARENT_SECTION, and until now NO Cypher pattern
# used it — the generator saw the edge in its schema, had no pattern telling it
# what to do with it, and fell back to a section_title regex. Measured on a
# 6-question pilot: every question fell back to Pattern B or the Pattern C
# sentinel, KG scored 0.111, and a hand-written tree query on the same six
# scored 0.633.
#
# The parent clause is resolved HERE against the KG and handed to the prompt
# verbatim, exactly as Pattern V hands over a Concept name. Letting the LLM
# invent the topic string is what produced 'activation' in that pilot — a phrase
# matching clauses all over the corpus, which buried the gold below rank 5.

# Generic titles that head a clause tree in dozens of specifications. Anchoring
# on one of these retrieves an arbitrary spec's subtree, so they are only usable
# when the question also pins the specification.
_HIER_GENERIC_TITLES = {
    "general", "description", "introduction", "scope", "overview", "definitions",
    "abbreviations", "requirements", "general requirements", "security",
    "charging", "activation", "deactivation", "registration", "invocation",
    "interrogation", "provision", "normal operation", "procedures",
    "security requirements", "high level requirements", "general aspects",
}
# A parent must head at least this many normative children to be worth gathering:
# below it the question is answerable from one clause and Pattern A/B is better.
_HIER_MIN_CHILDREN = 3
# and the phrase must not match more parents than this, or it is a generic title
# wearing a specific-looking name.
_HIER_MAX_PARENTS = 4

_HIER_CYPHER = """
UNWIND $phrases AS ph
MATCH (p:Chunk) WHERE toLower(p.section_title) CONTAINS ph
  AND ($specs = [] OR p.spec_id IN $specs)
MATCH (c:Chunk)-[:PARENT_SECTION]->(p)
WHERE toLower(c.content) CONTAINS 'shall'
WITH ph, p, count(DISTINCT c) AS n_kids
WHERE n_kids >= $min_kids
WITH ph, collect({chunk_id: p.chunk_id, spec_id: p.spec_id,
                  title: p.section_title, n_kids: n_kids}) AS parents
WHERE size(parents) <= $max_parents
UNWIND parents AS p
RETURN ph AS phrase, p.chunk_id AS chunk_id, p.spec_id AS spec_id,
       p.title AS title, p.n_kids AS n_kids
ORDER BY p.n_kids DESC
LIMIT 3
"""


def _hier_phrases(question: str) -> list[str]:
    """Content-word n-grams of the question, longest first.

    Longest-first matters: 'localised service area' must be tried before
    'service area', which heads clause trees in unrelated specifications.
    """
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9/_-]*", (question or "").lower())
    stop = {"the", "a", "an", "of", "for", "and", "or", "to", "in", "on", "at",
            "is", "are", "what", "which", "does", "do", "how", "we", "our", "it",
            "that", "this", "with", "from", "by", "as", "when", "must", "shall"}
    out = []
    for n in (6, 5, 4, 3, 2):
        for i in range(len(words) - n + 1):
            gram = words[i:i + n]
            if gram[0] in stop or gram[-1] in stop:
                continue
            phrase = " ".join(gram)
            if phrase not in _HIER_GENERIC_TITLES:
                out.append(phrase)
    return out[:40]


def detect_section_hierarchies(driver, question: str,
                               spec_refs: Optional[list] = None) -> list[dict]:
    """Parent clauses the question names whose normative content sits in children.

    Returns at most one entry (the richest subtree) so the prompt stays a menu
    rather than a list of alternatives to weigh. Empty when nothing distinctive
    matches — which is the common case and must stay cheap, hence the single
    round-trip over pre-filtered phrases.
    """
    phrases = _hier_phrases(question)
    if not phrases or driver is None:
        return []
    specs = [s for s in (spec_refs or []) if s]
    try:
        with driver.session() as s:
            rows = s.run(_HIER_CYPHER, phrases=phrases, specs=specs,
                         min_kids=_HIER_MIN_CHILDREN,
                         max_parents=_HIER_MAX_PARENTS).data()
    except Exception:
        return []
    if not rows:
        return []
    # Longest matching phrase wins, then the richest subtree: a question naming
    # 'network personalisation' must not anchor on a bare 'personalisation'
    # parent that also heads the SP and corporate variants.
    rows.sort(key=lambda r: (len(r["phrase"]), r["n_kids"]), reverse=True)
    return rows[:1]
