"""
RAG pipeline orchestrator — coordinates retrieval stages and yields SSE events.

Modes (selected per request):
  - 'fixed':       intent → vector → graph (LLM Cypher) → rrf fusion → rerank → answer
  - 'hybrid':      same as 'fixed' but SKIPS the final cross-encoder rerank — ablation
                            mode isolating the reranker's own contribution (thesis Table 3.3)
  - 'react_agent': intent → ADAPTIVE ReAct (LLM picks vector/cypher/expand_term/
                            inspect_chunk/finish each iter) → rerank → answer
  - 'vector_only' / 'kg_only' / 'llm_only': single-branch ablations, see Step 2 below
"""
import asyncio
import os
import re
from collections.abc import AsyncIterator, Iterator
from typing import Any


# Wrap a sync iterator (Ollama stream, graph search) as async so the event loop
# is free between each `next()` — uvicorn needs to tick the loop to flush SSE
# bytes to the socket. Without this the orchestrator blocks ~15-30s and the WHOLE
# stream reaches the client only once at the end (uvicorn can't flush while the
# event loop is busy).
#
# Important: do NOT raise StopIteration across an asyncio Future (Python forbids
# it — `TypeError: StopIteration interacts badly with generators`). Use a sentinel
# returned from the sync helper to avoid that.
_ITER_DONE = object()

# Grounding rule #3 (llm/prompts.py) tells the model to write "Context does
# not cover <sub-question>." (or "does not specify") when nothing in the
# retrieved chunks supports an answer. When that line IS the whole answer
# (no citation bracket anywhere → nothing was actually grounded), the
# reranked chunks are noise the user never sees the answer point to — so we
# suppress the sources list for that case instead of showing them anyway.
_NO_COVERAGE_RE = re.compile(r"^context does not (cover|specify)", re.IGNORECASE)


def _answer_has_no_coverage(answer_text: str) -> bool:
    stripped = answer_text.strip()
    return bool(_NO_COVERAGE_RE.match(stripped)) and "[" not in stripped


# Ollama's think/no-think toggle only splits the `thinking` field on /api/chat;
# our client uses /api/generate (see ollama_client.py), where reasoning models
# (verified live: deepseek-r1:14b, Ollama 0.32.3) emit the reasoning trace as
# literal <think>...</think> text INSIDE `response` regardless of `think:false`.
# Strip it defensively so it never leaks into the answer users/callers see.
# No-op when Ollama does split channels correctly (nothing to strip then).
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _strip_inline_think(answer_text: str) -> str:
    return _THINK_BLOCK_RE.sub("", answer_text).strip()


def _next_or_done(it: Iterator[Any]) -> Any:
    try:
        return next(it)
    except StopIteration:
        return _ITER_DONE


async def _async_iter(sync_iter: Iterator[Any]) -> AsyncIterator[Any]:
    while True:
        ev = await asyncio.to_thread(_next_or_done, sync_iter)
        if ev is _ITER_DONE:
            break
        yield ev


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _safe_flush_len(buf: str, marker: str) -> int:
    """How many leading chars of `buf` are safe to flush now — i.e. the remaining
    tail cannot be the start of `marker` arriving split across future chunks.
    Keeps the longest suffix of `buf` that's still a valid prefix of `marker`."""
    for keep in range(min(len(marker) - 1, len(buf)), 0, -1):
        if marker.startswith(buf[-keep:]):
            return len(buf) - keep
    return len(buf)


async def _split_inline_think(events: AsyncIterator[dict]) -> AsyncIterator[dict]:
    """Re-tag a literal leading <think>...</think> block inside "response" events
    as "thinking" events, for the Ollama /api/generate + reasoning-model combo that
    doesn't split the channel natively (see _strip_inline_think). Buffers just
    enough of the stream to decide whether it opens with the tag — resolves after
    a handful of tokens for a normal answer, so streaming latency is unaffected.
    Already-tagged "thinking" events pass through untouched."""
    buf = ""
    in_think = False
    resolved = False  # True once we know this stream does/doesn't open with <think>
    async for ev in events:
        if ev["kind"] != "response" or resolved:
            yield ev
            continue
        buf += ev["token"]
        if not in_think:
            stripped = buf.lstrip()
            if stripped.startswith(_THINK_OPEN):
                in_think = True
                buf = stripped[len(_THINK_OPEN):]
            elif _THINK_OPEN.startswith(stripped):
                continue  # still an ambiguous prefix (e.g. "<thi") — wait for more
            else:
                resolved = True
                yield {"kind": "response", "token": buf}
                buf = ""
            continue
        idx = buf.find(_THINK_CLOSE)
        if idx == -1:
            # Closing tag not (yet) fully in buf — it may be split across the next
            # chunk (e.g. "</th" + "ink>"), so hold back a possible-prefix tail
            # instead of flushing the whole buffer (which would destroy the tag).
            flush_len = _safe_flush_len(buf, _THINK_CLOSE)
            if flush_len:
                yield {"kind": "thinking", "token": buf[:flush_len]}
            buf = buf[flush_len:]
        else:
            think_part = buf[:idx]
            if think_part:
                yield {"kind": "thinking", "token": think_part}
            rest = buf[idx + len(_THINK_CLOSE):].lstrip()
            in_think = False
            resolved = True
            buf = ""
            if rest:
                yield {"kind": "response", "token": rest}
    if buf:
        yield {"kind": "thinking" if in_think else "response", "token": buf}


from neo4j import GraphDatabase

from retrieval.step_evidence import (
    STEP_EVIDENCE, attach_step_evidence, mark_step_actor_chunks,
)
from pipeline import ablations as ABL
from retrieval.title_search import (
    TitleSearcher,
    TITLE_SEARCH_OFF,
    TOP_K_TITLE,
    rrf_merge as _title_rrf_merge,
)
from retrieval import (
    VectorSearcher,
    BM25Searcher,
    GraphSearcher,
    AdaptiveHopSearcher,
    rrf_fusion,
    rerank,
    rerank_per_gap,
    LLMCypherGenerator,
)
from retrieval.anchor_retrieval import detect_value_concepts, fetch_anchor_chunks
from retrieval.reference_hop import expand_reference_chain
from retrieval.triple_serialize import serialize_triples
from llm import OllamaClient, build_prompt, build_llm_only_prompt
from llm.model_profiles import resolve_num_ctx
from pipeline.intent_classifier import IntentClassifier
from pipeline.term_first import TermFirstStrategy
from pipeline.term_index import build_term_index
from pipeline.scope_check import scope_warnings

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
# Model used by graph-search Cypher generation AND the adaptive planner.
CYPHER_MODEL = os.getenv("CYPHER_MODEL", "qwen3:14b")

# Adaptive ReAct retrieval (react_agent mode only). Quality > latency per project lead.
ADAPTIVE_HOP_MAX_ITER = int(os.getenv("ADAPTIVE_HOP_MAX_ITER", "5"))
# 0 (or any non-positive value) disables the wall-clock budget — only iter cap applies.
ADAPTIVE_HOP_BUDGET_MS = int(os.getenv("ADAPTIVE_HOP_BUDGET_MS", "0"))
ADAPTIVE_HOP_PLANNER_MODEL = os.getenv("ADAPTIVE_HOP_PLANNER_MODEL", CYPHER_MODEL)

# Retrieval widths are TUNING knobs, not deployment config: they belong on the
# command line of the run being measured, never in `.env`. `main.py` calls
# load_dotenv() at import, and although that does not overwrite a variable the
# process already carries, relying on that precedence silently couples an
# experiment to a file someone else may edit. `_tune()` therefore reads the value
# captured at interpreter start (before load_dotenv ran) and ignores `.env`
# entirely, so
#     TOP_K_GRAPH=14 uvicorn main:app ...
# is the only way to change one.
def _proc_env() -> dict:
    """The process environment as it was BEFORE main.py loaded .env.

    Falls back to os.environ when the orchestrator is imported outside the API
    (tests, scripts) — there is no .env pollution in that case."""
    try:
        import main  # noqa: PLC0415 - deliberate late import, see docstring
        return getattr(main, "_PROC_ENV", os.environ)
    except Exception:
        return os.environ


def _tune(name: str, default: int) -> int:
    raw = _proc_env().get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# Both branches enter RRF at the SAME width (2026-08-29). `TOP_K_GRAPH = 8`
# against `TOP_K_VECTOR = 10` was inherited from the first commit and had never
# been measured — every other width here (WIDE, FINAL, RESERVED) carries a dated
# ablation note, that pair did not. Measured on the 100 procedure questions of
# kg_bench_300_v2, `fixed` mode: making the two widths equal AND dropping
# VECTOR_RESERVED_SLOTS to 0 lifts Recall@5 from 0.521 to 0.627 (KG-only, the
# ceiling for this class, is 0.655) — it closes 79% of the gap. Recall@10 is
# unchanged (0.710 vs 0.709), which is the tell: the same chunks were being
# found either way, the old asymmetry merely kept them out of the top 5.
# They also size the candidate pool (TOP_K_VECTOR + TOP_K_GRAPH below), not just
# the mix ratio. Env-overridable to restore the old 10/8 split for comparison
# with runs up to v48.
TOP_K_VECTOR = _tune("TOP_K_VECTOR", 10)
TOP_K_GRAPH = _tune("TOP_K_GRAPH", 10)
# Graph branch retrieves WIDE then reranks internally (2026-07-12, options 2+3
# from BAO_CAO_5_MODE.md). Pattern F/G emit a constant `score`, so the in-Cypher
# `LIMIT $top_k` cut chunks in arbitrary Neo4j order — gold routinely fell
# outside the top-8 even when the query DID match it (G-001: 30 matches contained
# all 3 gold, LIMIT 8 kept none). We now run the Cypher with a WIDE limit
# (TOP_K_GRAPH_WIDE) so the gold is in the result set, then let a cross-encoder
# rerank pick the real top-TOP_K_GRAPH by relevance before RRF — replacing the
# arbitrary LIMIT with a quality gate. Env-overridable for tuning.
# Bumped 40→100 (2026-07-12, "AMF↔UDM registration" case): with real per-chunk
# FG scores the gold registration chunk (ts_23_502_4.2.2.2.2 — a `definition`
# chunk with NO Step, so structurally lower than 60+ procedure chunks that share
# the same actor pair) sat at rank ~63 of 344 matches. A width of 40 still cut it;
# 100 keeps it in the pool for the internal cross-encoder rerank to lift by
# content relevance. The rerank down to TOP_K_GRAPH is the real quality gate.
TOP_K_GRAPH_WIDE = _tune("TOP_K_GRAPH_WIDE", 40)
# Bumped 6→8 (2026-07-07 ablation study, tests/benchmark/wh_multigold_100_v2/
# BAO_CAO_5_MODE.md): multi-gold questions carry up to 5 gold chunks each, and
# 46% of zero-recall cases were "near-miss" — right section family, wrong
# sub-clause, simply because there wasn't room left in top-6 after the general
# candidate ranking. The extra headroom is mostly consumed by neighbor
# expansion below, not by widening the base rerank selection.
# Bumped 8→10 (2026-08-11): the paper Retrieval / KG-Ablation sheets report R@10,
# which is undefined when the final ranked list is only 8 long. Env-overridable so
# an 8-chunk context can be restored for comparison with the earlier runs.
TOP_K_FINAL = _tune("TOP_K_FINAL", 10)
# How many of the top vector hits are guaranteed a slot in the final answer
# context, protected from being evicted by the graph branch flooding the RRF pool
# (added 2026-07-12 at 3, wh_kg_v3_100 F/G/O regression — see Step 3a-vec).
# TURNED OFF 2026-08-29. The reservation only ever applied to procedure /
# relationship / interface intents, and on the 100 procedure questions of
# kg_bench_300_v2 that is exactly where it cost the most: the three protected
# slots were taken out of a 10-slot context on questions carrying 3.94 gold
# chunks on average, so the branch that had actually found the gold cluster could
# not seat it. Dropping it to 0 together with the equal RRF widths above moved
# Recall@5 0.521 -> 0.627 against a 0.655 KG-only ceiling. The regression it was
# introduced for is a top-1/top-2 vector case, which the reranker now seats on
# merit; set it back to 3 if that case reappears.
VECTOR_RESERVED_SLOTS = _tune("VECTOR_RESERVED_SLOTS", 0)
# Interface / reference-point identifier in the question (N1..N99, S1, S5, Xn, Uu,
# X2). These questions use Pattern B (section_title regex) whose flat-score flood
# evicts the vector gold — protect vector slots for them too (see Step 3a-vec).
_INTERFACE_ID_RE = re.compile(r"\b(N\d{1,2}|S\d{1,2}|Xn|X2|Uu)\b\s*(interface|reference)", re.IGNORECASE)

# When the graph branch is empty, force rerank to keep at least N chunks in
# context (floor) so the answer LLM doesn't run on empty context. Filtering
# off-topic chunks (out-of-domain questions) is done AT THE FUSION LAYER via a
# logit-band — see fusion._rerank_with_dedup (the cross-encoder logit separates
# in/out-domain very cleanly, cosine does NOT).
RERANK_GRAPH_EMPTY_MIN_KEEP = int(os.getenv("RERANK_GRAPH_EMPTY_MIN_KEEP", "2"))

# Graph chunks below this LLM-emitted confidence score (see cypher_generator.py's
# Pattern A CASE ladder: 1.0 exact title / 0.92 MENTIONS / 0.85-0.75 chunk_type /
# 0.6 ELSE-fallback) are dropped before RRF fusion. The 0.6 tier means "chunk is
# in the right spec but nothing more specific matched" — exactly the "exists but
# not relevant" noise identified in the kg_only ablation (BAO_CAO_5_MODE.md
# §5.1: 38/100 non-empty graph results, but most didn't survive rerank because
# they were topically off). Filtering here keeps that noise out of the RRF pool
# entirely, instead of relying on rerank to filter it after the fact.
GRAPH_MIN_SCORE = 0.75

# Neighbor expansion (sibling clauses) — see _expand_neighbors().
#
# Raised 2 -> 10 on 2026-08-25 after measuring the aggregation class: questions whose
# answer is spread over 3-5 sibling clauses cannot be completed from two extra chunks
# no matter how good the seed is. On 50 such questions (tests/benchmark/kg_req_100
# cells 2 and 4, seeds = the dense branch's top-5) Recall@10 went 0.244 -> 0.388 and
# Recall@15 0.244 -> 0.402, while the 50 single-clause questions in the same set moved
# 0.760 -> 0.780, i.e. no regression on the class this cap was originally tuned for.
# Recall@5 is unchanged either way — the recovered siblings land below rank 5, so this
# only pays off when TOP_K_FINAL is large enough to keep them.
NEIGHBOR_MAX_EXTRA = _tune("NEIGHBOR_MAX_EXTRA", 10)

# Dynamic RRF weight for the graph branch (2026-07-09, "fixed vs vector_only"
# ablation follow-up — tests/benchmark/wh_multigold_100_v2_after_fixes/). A
# flat weights=[1.0, 1.0] treats every graph hit as equally trustworthy as a
# vector hit regardless of HOW it matched. `score`'s CASE-ladder tier (see
# GRAPH_MIN_SCORE above) already tells us that: >=0.92 means an exact
# section-title match or a MENTIONS edge fired (Pattern A/B — a real anchor),
# while the 0.75-0.92 band is the tier that only just cleared the noise filter
# (chunk_type match alone). Trust the graph branch MORE than vector when it
# found a real anchor, LESS when it only barely qualified — instead of always
# splitting the vote 50/50.
GRAPH_HIGH_CONF = 0.92
GRAPH_HIGH_CONF_WEIGHT = 1.4
GRAPH_LOW_CONF_WEIGHT = 0.6

# Flat-score guard (2026-07-12, wh_kg_v3_100 5-mode ablation — BAO_CAO_5_MODE.md).
# Pattern F/G emit a CONSTANT `score` (1.0 / 0.9) on EVERY chunk they return —
# unlike Pattern A's CASE-ladder where the score is a real per-chunk confidence.
# When every graph chunk carries the same score, that score is NOT a confidence
# signal: it just means "the pattern matched", not "this chunk is the answer".
# The old _graph_rrf_weight read peak-score >=0.92 and handed Pattern F its full
# 1.4 trust weight, so 8 undifferentiated (often gold-free, because the in-Cypher
# LIMIT cut randomly) graph chunks out-ranked the vector branch in RRF and evicted
# the vector hits that actually held the gold (measured: fixed 0.335 < vector
# 0.361; 6 questions where fixed lost gold the vector branch had at rank 1-4).
# Fix: when the graph scores are flat, treat the branch as LOW confidence —
# let the cross-encoder rerank arbitrate instead of trusting the RRF weight.
GRAPH_FLAT_SCORE_WEIGHT = 0.5

# Spec-type bias for definition/functionality questions (2026-07-12, "SCP
# functionalities" case). Pattern A matches `c.spec_id IN t.source_specs`, which
# pulls EVERY chunk of a spec that lists the term — including Scope/References/
# Abbreviations that merely name it. A TR study (e.g. TS 23.783) that discusses
# the term densely then swamps the real functional chunks in the normative TS
# (23.501 architecture, 29.500 SBI), so the answer cites the study instead of the
# spec. Nudge the graph `score` (rerank reads it as `_BLEND_BETA*upstream`, so a
# small factor tilts rank without overriding the cross-encoder logit): demote TR
# study specs, promote chunks whose section actually describes the function.
# TR series: the 2nd spec number ≥ 700 marks a Technical Report (study), e.g.
# ts_23_783, ts_28_840 — normative TS specs use lower section series.
# Extended to NF↔NF interaction intents (2026-07-12, "AMF↔UDM registration"
# case): a `relationship`/`procedure`/`how_does` question over two NFs pulls in
# TR-study procedure chunks (ts_23_700-41/08 "Registration") that a wide FG match
# surfaces alongside the normative TS 23.502 flow — and the reranker put the TRs
# on top (rerank_score 4.80 > TS 4.25). Demoting the study specs here lets the
# normative TS registration chunk win the RRF/rerank blend.
_SPEC_BIAS_INTENTS = {
    "definition", "what_is", "list", "capability",
    "relationship", "procedure", "how_does",
}
# A/B kill-switches for the 2026-07-12 retrieval changes (regression bisect on
# wh_kg_v3_100: fixed R@5 dropped 0.423→0.282 vs the mph baseline). Each defaults
# ON; set the env to 0 to disable that component and isolate the culprit.
ANCHOR_ON = os.getenv("ANCHOR_ON", "1") not in ("0", "false", "")
PROC_FRONT_ON = os.getenv("PROC_FRONT_ON", "1") not in ("0", "false", "")
# Evidence format fed to the answer LLM. "text" (default) = chunk prose with
# [spec §section] citations. "triples" = serialized KG triples of the retrieved
# chunks' neighbourhood — the paper's Study-Ablation "KG-only (KG triples)" row.
KG_EVIDENCE = os.getenv("KG_EVIDENCE", "text").lower()
# Provenance ablation: when "0", strip the [spec §section] citation line from the
# context so the "Text+KG WITHOUT provenance" Study-Ablation row can be measured.
PROVENANCE_ON = os.getenv("PROVENANCE_ON", "1") not in ("0", "false", "")
# Reranker ablation for modes other than `hybrid`. The Study-Ablation ladder used
# to be uninterpretable because its first two rows were reranked and its middle two
# were not — only the final step was a single-variable delta. This lets any mode be
# run unreranked so every step of the ladder changes exactly one thing.
RERANK_OFF = os.getenv("RERANK_OFF", "0") in ("1", "true", "True")
# Add the sparse branch to the fixed/hybrid fusion (see the fusion site below).
BM25_BRANCH = os.getenv("BM25_BRANCH", "0") in ("1", "true", "True")

# Refuse straight away when retrieval came back empty, instead of prompting the LLM
# with no context. Set to 0 to reproduce a measurement taken before this existed —
# it changes both the answer text and the latency of every empty-retrieval question.
ABSTAIN_ON_EMPTY = os.getenv("ABSTAIN_ON_EMPTY", "1") not in ("0", "false", "")

# KG-layer ablation (paper "KG Ablation" sheet). Cumulative: each level adds ONE
# edge family on top of the level below, so the row-to-row delta is that layer's
# contribution. Meant to be run in kg_only mode.
#   entity     — bare Chunk-[:MENTIONS]->Term lookup only (weakest KG signal)
#   relation   — + graph-Cypher traversal (INVOLVES/CO_OCCURS/PROVIDED_BY/DESCRIBES_*) + anchors
#   hierarchy  — + PARENT_SECTION neighbour expansion
#   xref       — + REFERENCES_CHUNK reference-hop
#   provenance — + [spec §section] citations (answer-side; recall unchanged, F1/faith move)
# "off" (default) disables the gate entirely → the live system is unchanged.
_KG_ABL_ORDER = ["entity", "relation", "hierarchy", "xref", "provenance"]
_kg_abl_raw = os.getenv("KG_ABLATION", "off").lower()
KG_ABL_LEVEL = _KG_ABL_ORDER.index(_kg_abl_raw) if _kg_abl_raw in _KG_ABL_ORDER else None


def _kg_layer_on(feature: str) -> bool:
    """Per-request ablation gate. Delegates to pipeline.ablations so the level can be
    set by the caller instead of only by the process environment — the env value is
    still the default when a request does not override it."""
    return ABL.kg_layer_on(feature)
SPEC_BIAS_ON = os.getenv("SPEC_BIAS_ON", "1") not in ("0", "false", "")
PROC_GEN_BIAS_ON = os.getenv("PROC_GEN_BIAS_ON", "1") not in ("0", "false", "")
TR_STUDY_PENALTY = float(os.getenv("TR_STUDY_PENALTY", "0.7"))
FUNCTIONAL_SECTION_BOOST = float(os.getenv("FUNCTIONAL_SECTION_BOOST", "1.15"))
_TR_SPEC_RE = re.compile(r"^ts_\d{2}_(\d{3})")
_FUNCTIONAL_SECTION_KW = (
    "requirement", "functional", "description", "general", "overview",
    "capabilit", "service", "role",
)


def _is_tr_study_spec(spec_id: str) -> bool:
    """TR study (Technical Report) — 2nd spec-number series ≥ 700 (e.g. 23.783,
    28.840). These discuss features as studies, not normative behaviour."""
    m = _TR_SPEC_RE.match(spec_id or "")
    return bool(m) and int(m.group(1)) >= 700


def _spec_type_bias(chunks: list[dict], resolved: dict | None) -> list[dict]:
    """Demote TR-study chunks and promote functional-section chunks for
    definition/functionality questions. Multiplies `score` in place-free copies;
    only fires for the biased intents (caller gates on intent). Full-name / kw
    match in section_title marks a chunk that actually describes the function
    (not Scope/References)."""
    full_names = [
        (v.get("full_name") or "").lower()
        for v in (resolved or {}).values() if v.get("full_name")
    ]
    out = []
    for c in chunks:
        factor = 1.0
        if _is_tr_study_spec(c.get("spec_id", "")):
            factor *= TR_STUDY_PENALTY
        title = (c.get("section") or c.get("section_title") or "").lower()
        if title and (
            any(fn and fn in title for fn in full_names)
            or any(kw in title for kw in _FUNCTIONAL_SECTION_KW)
        ):
            factor *= FUNCTIONAL_SECTION_BOOST
        out.append({**c, "score": (c.get("score") or 0) * factor} if factor != 1.0 else c)
    return out


# Generality bias for procedure/interaction questions (2026-07-12, "AMF↔UDM
# registration" case). 3GPP procedures come in a canonical "General <X>" chunk
# plus narrower variant chunks ("Registration WITH AMF re-allocation", "…WITH
# Onboarding SNPN", "…for non-3GPP access"). A question about the general
# interaction wants the canonical chunk, but the cross-encoder scores the variant
# ≈ the general one (both are "registration"). Nudge the canonical up and the
# variant down so the base procedure wins the tie. Signal is purely lexical on the
# section title — no content scan.
# Only procedure/interaction intents want the general-vs-variant nudge — a
# definition question has no "General <X>" canonical to prefer.
_PROC_BIAS_INTENTS = {"relationship", "procedure", "how_does"}

# Reference-chain hop expansion (2026-07-25, follow-up to de_cuong/
# PhuLuc_B_5ca_vector_vs_fixed.md). That report measured 5 questions where the
# answer chunk sits one REFERENCES_CHUNK hop (cross-spec) from a chunk
# retrieval DOES find, and neither `vector_only` nor `fixed` recovered it —
# every existing Cypher pattern (A1/A2/B/F/G/M/P) re-anchors on the question's
# terms/steps, none follow "this chunk cites that chunk, go get it." Gated to
# intents where a citation chain is plausible: relationship/procedure/how_does
# (already the anchor-retrieval set) PLUS definition (Ca 3 in the report —
# "what does the SoR container contain" — was intent=definition; the container
# structure was defined in a DIFFERENT spec than the one describing when it's
# sent). `network_function`/`comparison`/`general` are excluded — a listing or
# comparison question isn't chasing a citation for one specific detail.
_HOP_INTENTS = _PROC_BIAS_INTENTS | {"definition"}
HOP_EXPANSION_ON = os.getenv("HOP_EXPANSION_ON", "1") not in ("0", "false", "")
# How many of the current best candidates (post RRF, pre-final-rerank) to use
# as hop seeds. Small on purpose — keeps the Cypher UNWIND narrow (this is what
# makes it safe to run on every gated question, unlike Term-seeded multi-hop
# which fans out to every chunk in a Term's source_specs).
HOP_SEED_TOP_N = int(os.getenv("HOP_SEED_TOP_N", "12"))
# Wide on purpose, same "retrieve wide, let cross-encoder rerank narrow"
# philosophy as TOP_K_GRAPH_WIDE above — a tight cap here reintroduces the
# exact arbitrary-order-drops-the-gold failure this codebase already fixed for
# Pattern F/G (round-robin-by-seed in reference_hop.py softens this further,
# but a low cap still risks losing seeds beyond the round-robin's reach).
HOP_MAX_TARGETS = int(os.getenv("HOP_MAX_TARGETS", "40"))

GENERAL_PROC_BOOST = float(os.getenv("GENERAL_PROC_BOOST", "1.12"))
VARIANT_PROC_PENALTY = float(os.getenv("VARIANT_PROC_PENALTY", "0.88"))
# Variant markers: a scenario-narrowing qualifier in the title. " with " catches
# "Registration with AMF re-allocation / Onboarding SNPN"; the rest catch common
# access/roaming/disaster narrowings.
_VARIANT_TITLE_KW = (
    " with ", "re-allocation", "non-3gpp", "untrusted", "roaming", "disaster",
    "onboarding", "emergency", "interworking", " via ",
)


def _procedure_generality_bias(chunks: list[dict]) -> list[dict]:
    """Promote the canonical "General <procedure>" chunk over its scenario-narrowed
    variants for procedure/interaction questions. Title starting with 'general' →
    boost; title carrying a variant qualifier → penalty. Score-multiplier, copies."""
    out = []
    for c in chunks:
        title = (c.get("section") or c.get("section_title") or "").lower().strip()
        factor = 1.0
        if title.startswith("general"):
            factor *= GENERAL_PROC_BOOST
        elif any(kw in title for kw in _VARIANT_TITLE_KW):
            factor *= VARIANT_PROC_PENALTY
        out.append({**c, "score": (c.get("score") or 0) * factor} if factor != 1.0 else c)
    return out


# Question-aware procedure boost (2026-07-12, four-round AMF↔UDM review). The
# root cause the reviewer flagged across all four runs: the cross-encoder cutting
# 100 graph chunks → TOP_K_GRAPH=8 has NO signal for "this chunk is the core
# <procedure> the question asked about" vs "this chunk merely contains both
# entities in an unrelated flow (reachability/provisioning/hosting-network)". The
# gold registration chunks (ts_23_502_4.2.2.2.2 General Registration rank 64,
# 4.13.3.1 SMS-over-NAS Registration rank 22) name the procedure in their
# section_title; the off-topic chunks do not. We extract the procedure keyword
# from the QUESTION and hard-protect graph chunks whose section_title matches it
# from the internal-rerank cut, so a relevant registration chunk cannot be evicted
# by a higher cross-encoder logit on an off-topic chunk.
# Cap on how many procedure-matched chunks bypass the cut — keeps a broad
# keyword ("registration" matches ~20 chunks) from flooding the graph pool.
PROC_PROTECT_MAX = int(os.getenv("PROC_PROTECT_MAX", "4"))
_PROCEDURE_KEYWORDS = (
    "registration", "deregistration", "authentication", "handover",
    "subscription", "selection", "establishment", "modification", "release",
    "reallocation", "reachability", "configuration update", "service request",
    "paging", "mobility", "provisioning",
)


def _question_procedure_keywords(question: str) -> list[str]:
    """Procedure keywords the question is actually about — used to protect the
    matching graph chunks from the internal-rerank cut."""
    q = (question or "").lower()
    return [kw for kw in _PROCEDURE_KEYWORDS if kw in q]


def _procedure_matched_ids(chunks: list[dict], keywords: list[str]) -> set[str]:
    """chunk_ids whose section_title matches a question procedure keyword. These
    are hard-protected through the internal rerank so a core-procedure chunk is
    never evicted by an off-topic chunk with a higher cross-encoder logit."""
    if not keywords:
        return set()
    matched = set()
    for c in chunks:
        title = (c.get("section") or c.get("section_title") or "").lower()
        cid = c.get("chunk_id")
        if cid and any(kw in title for kw in keywords):
            matched.add(cid)
    return matched


# BM25 branch scores enter the rerank blend as `upstream` (fusion._rerank_with_dedup),
# which assumes [0,1] like a dense cosine or a normalised graph score. Raw Lucene
# scores (13-60) let 0.3*upstream outweigh the cross-encoder and push every BM25 hit
# over the canonical threshold (upstream >= 1 -> floor -2). ON by default since
# 2026-10-06: the paper's BM25 configs (v53/v54/v59, TeleQnA) were re-measured with it;
# BM25_SCORE_NORM=0 on the command line restores the raw-score behaviour of the runs
# before that date.
# The cap keeps the top hit below the canonical threshold, as a dense hit always is.
_BM25_NORM_CAP = 0.99


def _bm25_norm_on() -> bool:
    return _proc_env().get("BM25_SCORE_NORM", "1").strip() not in ("0", "false", "False")


def _normalise_bm25_scores(chunks: list[dict]) -> list[dict]:
    """Scale BM25 scores by the per-query maximum into [0, _BM25_NORM_CAP], keeping
    their order. Returns new dicts (does not mutate)."""
    if not chunks:
        return chunks
    mx = max((c.get("score") or 0) for c in chunks)
    if mx <= 0:
        return chunks
    return [{**c, "score": (c.get("score") or 0) / mx * _BM25_NORM_CAP} for c in chunks]


def _normalise_graph_scores(chunks: list[dict]) -> list[dict]:
    """Rescale graph `score` into [0,1] when it exceeds 1 (Pattern F/G emit raw
    counts like n_steps=13 / n_via_op=2). Divide by the max so the top chunk is
    1.0 and the rest keep their relative rank, on the same scale as Pattern A's
    0..1 CASE-ladder. No-op when scores are already ≤1 (Pattern A/B/D) so their
    calibrated thresholds are untouched. Returns new dicts (does not mutate)."""
    if not chunks:
        return chunks
    mx = max((c.get("score") or 0) for c in chunks)
    if mx <= 1.0 or mx == 0:
        return chunks
    return [{**c, "score": (c.get("score") or 0) / mx} for c in chunks]


def _filter_low_confidence_graph(chunks: list[dict]) -> list[dict]:
    """Drop graph chunks whose LLM-emitted `score` is below GRAPH_MIN_SCORE —
    see GRAPH_MIN_SCORE docstring above for why this tier is noisy."""
    return [c for c in chunks if (c.get("score") or 0) >= GRAPH_MIN_SCORE]


def _bare_term_retrieval(driver, abbrevs: list[str], top_k: int) -> list[dict]:
    """Weakest KG signal: chunks that MENTION the question's resolved Term nodes,
    ranked by how many distinct asked terms they co-mention. This is the "Entity
    only" layer of the KG-ablation (Chunk-[:MENTIONS]->Term, no relation traversal,
    no anchors) — a bare entity lookup, deliberately much weaker than the graph
    query layer so the ablation's row-to-row delta is meaningful."""
    abbrevs = [a for a in (abbrevs or []) if a]
    if not abbrevs:
        return []
    cypher = """
    UNWIND $terms AS ab
    MATCH (c:Chunk)-[:MENTIONS]->(t:Term {abbreviation: ab})
    WITH c, count(DISTINCT t) AS score
    RETURN c.chunk_id AS chunk_id, c.content AS content, c.spec_id AS spec_id,
           c.section_title AS section, score
    ORDER BY score DESC LIMIT $top_k
    """
    with driver.session() as s:
        return [dict(r) for r in s.run(cypher, terms=abbrevs, top_k=top_k)]


def _graph_scores_are_flat(chunks: list[dict]) -> bool:
    """True when every graph chunk carries the same score — the Pattern F/G
    constant-score signature. In that case the score conveys no per-chunk rank
    information (see GRAPH_FLAT_SCORE_WEIGHT)."""
    scores = {round(c.get("score") or 0, 4) for c in chunks}
    return len(scores) <= 1


def _graph_rrf_weight(chunks: list[dict]) -> float:
    """Pick the graph branch's RRF weight from its confidence signal. Empty
    chunks → weight is moot, return 1.0. Flat scores (Pattern F/G) → LOW weight
    so undifferentiated graph hits can't evict real vector hits from the RRF
    pool. Otherwise (Pattern A CASE-ladder) use peak-score as before."""
    if not chunks:
        return 1.0
    if _graph_scores_are_flat(chunks):
        return GRAPH_FLAT_SCORE_WEIGHT
    max_score = max((c.get("score") or 0) for c in chunks)
    return GRAPH_HIGH_CONF_WEIGHT if max_score >= GRAPH_HIGH_CONF else GRAPH_LOW_CONF_WEIGHT


def _expand_neighbors(
    driver, chunk_ids: list[str], existing_ids: set[str], max_extra: int,
) -> list[dict]:
    """Pull sibling chunks (same parent section) of the given chunk_ids that
    aren't already in existing_ids. Multi-gold questions often need several
    adjacent sub-clauses of one feature (e.g. ts_23_401_4.13.2/.3/.4); rerank
    scores each chunk independently and can starve the rest of a coherent
    cluster even when one sibling already scored well.

    Two sources, in order, because NEITHER dominates the other and that was
    measured both ways:

      1. the `PARENT_SECTION` edge, restricted to siblings that carry normative
         language. Exact where it exists: on 50 aggregation questions every gold
         chunk had the edge and 47/50 had all their gold under ONE parent, and
         using the edge beat prefix matching by +0.085 Recall@10. The `shall`
         filter is load-bearing, not cosmetic — without it non-normative siblings
         fill the cap and push the gold back out (0.388 -> 0.345).
      2. `section_id` prefix matching (same spec_id, same parent prefix, same
         depth), as the fill. The edge covers only ~24% of chunks corpus-wide and
         returns ZERO siblings for real multi-gold clusters — re-verified here:
         ts_23_502_4.9.1.2.2 and ts_23_401_4.13.2 get 0 from the edge and 4 and 8
         from the prefix. Dropping the prefix would regress exactly those.

    Sync/blocking — call via asyncio.to_thread."""
    if not chunk_ids or max_extra <= 0:
        return []
    edge_query = """
    UNWIND $ids AS cid
    MATCH (c:Chunk {chunk_id: cid})-[:PARENT_SECTION]->(p:Chunk)
    MATCH (sib:Chunk)-[:PARENT_SECTION]->(p)
    WHERE toLower(sib.content) CONTAINS 'shall'
      AND NOT sib.chunk_id IN $existing
    RETURN DISTINCT sib.chunk_id AS chunk_id, sib.content AS content,
           sib.spec_id AS spec_id, sib.section_title AS section
    LIMIT $limit
    """
    prefix_query = """
    UNWIND $ids AS cid
    MATCH (c:Chunk {chunk_id: cid})
    WITH c, c.spec_id AS spec, split(c.section_id, '.') AS parts
    WHERE size(parts) > 1
    WITH DISTINCT spec, parts,
         reduce(s='', i IN range(0, size(parts)-2) | s + CASE WHEN s='' THEN '' ELSE '.' END + parts[i]) AS parent_sid
    MATCH (sib:Chunk)
    WHERE sib.spec_id = spec AND sib.section_id STARTS WITH parent_sid + '.'
      AND size(split(sib.section_id, '.')) = size(parts)
      AND NOT sib.chunk_id IN $existing
    RETURN DISTINCT sib.chunk_id AS chunk_id, sib.content AS content,
           sib.spec_id AS spec_id, sib.section_title AS section
    LIMIT $limit
    """
    out: list[dict] = []
    seen: set[str] = set(existing_ids)
    with driver.session(default_access_mode="READ") as session:
        for query in (edge_query, prefix_query):
            if len(out) >= max_extra:
                break
            rows = session.run(query, ids=chunk_ids, existing=list(seen),
                               limit=max_extra - len(out))
            for r in rows:
                d = dict(r)
                if d["chunk_id"] in seen:
                    continue
                seen.add(d["chunk_id"])
                out.append(d)
                if len(out) >= max_extra:
                    break
    return out


# Split a chunk's content into semantic segments at natural 3GPP boundaries,
# strongest first: numbered procedure steps ("1.\t", "3a.\t"), then blank-line
# paragraphs, then single lines. Adjacent tiny pieces are glued back so we don't
# over-fragment. Used to sub-chunk a long section and keep only the parts that
# actually answer the question (see _fit_chunk_content).
_STEP_SPLIT_RE = re.compile(r"(?=(?:\r?\n)\d+[a-z]?\.\t)")


def _semantic_segments(content: str, min_seg: int = 200) -> list[str]:
    if not content:
        return []
    # 1) numbered procedure steps — the natural unit of a call flow.
    segs = [s.strip() for s in _STEP_SPLIT_RE.split(content) if s.strip()]
    if len(segs) < 2:
        # 2) no steps → paragraphs (blank line), else single lines.
        raw = re.split(r"\n\s*\n", content) if "\n\n" in content else content.split("\n")
        segs = [s.strip() for s in raw if s.strip()]
    if len(segs) < 2:
        return [content.strip()]
    # Glue tiny fragments onto the previous segment so a lone heading/figure line
    # doesn't become its own segment.
    merged: list[str] = []
    for s in segs:
        if merged and len(merged[-1]) < min_seg:
            merged[-1] = merged[-1] + "\n" + s
        else:
            merged.append(s)
    return merged


def _fit_chunk_content(question: str, content: str, budget: int) -> str:
    """Return at most `budget` chars of `content`, choosing the parts most
    relevant to `question`. If the chunk fits, return it whole. Otherwise split
    it into semantic segments, cross-encoder-rerank the segments against the
    question, and greedily keep the highest-scoring ones (restored to original
    order) until the budget is hit — so the answer LLM sees the passages that
    actually address the question instead of a blind leading truncation."""
    content = content or ""
    if len(content) <= budget:
        return content
    segs = _semantic_segments(content)
    if len(segs) < 2:
        return content[:budget].rstrip() + "…"
    try:
        from models import get_reranker_model
        model = get_reranker_model()
        scores = model.predict([(question, s) for s in segs]).tolist()
    except Exception:
        # Reranker unavailable → fall back to leading truncation.
        return content[:budget].rstrip() + "…"
    # Greedily take segments by descending relevance, then re-emit in reading order.
    order = sorted(range(len(segs)), key=lambda i: scores[i], reverse=True)
    keep: set[int] = set()
    used = 0
    for i in order:
        seg_len = len(segs[i]) + 1  # +1 for the join newline
        if used + seg_len > budget:
            continue
        keep.add(i)
        used += seg_len
    if not keep:  # even the top segment is bigger than budget
        return segs[order[0]][:budget].rstrip() + "…"
    kept = [segs[i] for i in range(len(segs)) if i in keep]
    # Mark elision so the model knows the section is excerpted, not truncated mid-thought.
    return "\n[…]\n".join(kept)


def _chunks_to_context(
    chunks: list[dict], question: str = "", max_chars: int = 20000,
    per_chunk_chars: int = 4000, with_provenance: bool = True,
) -> tuple[str, list[str]]:
    """Format reranked chunks into the LLM context string with citations.

    Two guards so a few huge 3GPP procedure sections don't starve the rest:
    - per_chunk_chars: a single chunk contributes at most this many chars. One
      `ts_23_502` registration section is ~19.5k chars — uncapped it eats almost
      the whole 20k budget and every chunk after it is dropped, leaving the
      answer LLM with only ~2 chunks (the "answer too short / no context"
      symptom). Instead of a blind leading truncation we SEMANTICALLY sub-chunk
      the section and keep the segments most relevant to `question`
      (see _fit_chunk_content) — so the retained excerpt actually answers it.
    - SKIP (not break) an oversized block: if adding a chunk would blow the total
      budget, skip it and KEEP GOING — a later, smaller chunk can still fit,
      instead of cutting the context off at the first big one.
    """
    parts = []
    used: list[str] = []
    total = 0
    for c in chunks:
        # Step-scoped evidence (experiment, off by default) replaces the clause body
        # with just the steps that involve what was asked. It is already narrow, so
        # it bypasses _fit_chunk_content — re-cutting a 400-char excerpt by relevance
        # would only risk dropping the step the excerpt exists to deliver.
        scoped = c.get("step_evidence")
        content = scoped if scoped else _fit_chunk_content(
            question, c.get("content") or "", per_chunk_chars)
        # Provenance ablation: without the citation the LLM cannot attribute a
        # claim to a spec/section (Study-Ablation "Text+KG" vs "+provenance").
        if with_provenance:
            citation = f"[{c.get('spec_id', '?')} §{c.get('section', '?')}]"
            block = f"{citation}\n{content}"
        else:
            block = content
        if total + len(block) > max_chars:
            continue
        parts.append(block)
        used.append(c.get("chunk_id"))
        total += len(block)
    # Also return WHICH chunks made it in. The budget skip above means `reranked`
    # is an upper bound, not the evidence set: listing all of it as "sources"
    # credits the answer to chunks the model never saw.
    return "\n\n---\n\n".join(parts), used


# Compact per-chunk preview list for the trail UI
def _chunks_to_preview(
    chunks: list[dict], n: int = 3, preview_chars: int = 100, score_key: str = "score"
) -> list[dict]:
    previews = []
    for c in chunks[:n]:
        content = (c.get("content") or "").strip().replace("\n", " ")
        if len(content) > preview_chars:
            content = content[:preview_chars].rstrip() + "…"
        previews.append({
            # chunk_id is what makes a trace analysable offline: without it a saved
            # .stages.json can show WHICH spec the graph branch returned but not
            # whether it returned the gold chunk, so any "did the reranker drop the
            # gold or was the Cypher simply wrong" question needs a fresh live run.
            "chunk_id": c.get("chunk_id"),
            "spec_id": c.get("spec_id", "?"),
            "section": c.get("section", "?"),
            "score": c.get(score_key),
            "preview": content,
        })
    return previews


class RAGOrchestrator:
    def __init__(self):
        # Single shared Neo4j driver — used by the TermIndex builder, graph searcher,
        # and schema introspector.
        self._driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        # In-memory snapshot of all Term nodes (~31k). Replaces hard-coded
        # NETWORK_FUNCTIONS + per-query Neo4j round-trips. Built once here,
        # shared with TermFirstStrategy and AdaptiveHopSearcher.
        self._term_index = build_term_index(self._driver)
        self._term_first = TermFirstStrategy(index=self._term_index)

        # Vector searcher: used as a deterministic step in fixed mode, AND as a tool
        # the planner can call in react_agent mode.
        self._vector = VectorSearcher(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

        # BM25 (sparse) searcher — powers the `bm25` and `bm25_dense` retrieval
        # baselines (paper Retrieval sheet: BM25, Hybrid=BM25+Dense). Neo4j
        # full-text index; ensure_index self-heals it if a rebuild predates it.
        self._bm25 = BM25Searcher(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        self._bm25.ensure_index()

        # Section-title searcher — ranked replacement for Pattern B's regex reach
        # into c.section_title. Merged into the graph branch below, NOT a separate
        # retrieval mode: it is graph-side signal, so kg_only benefits from it.
        self._title = TitleSearcher(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        self._title.ensure_index()

        self._llm = OllamaClient()
        # LLM-first intent classifier (regex fallback). Uses the same OllamaClient
        # so Ollama keeps the model warm; classify calls with format=json + think=False.
        self._intent_clf = IntentClassifier(self._llm)
        self._cypher_gen = LLMCypherGenerator(self._llm, model=CYPHER_MODEL)
        # Graph search (LLM Cypher single-shot) — fixed mode only.
        self._graph = GraphSearcher(
            NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, generator=self._cypher_gen
        )

        # Adaptive ReAct retrieval — react_agent mode only. Picks one tool per
        # iteration (cypher_query / vector_search / expand_term / inspect_chunk /
        # finish) and decides on its own when there's enough evidence to stop.
        self._adaptive_hop = AdaptiveHopSearcher(
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD,
            llm=self._llm,
            planner_model=ADAPTIVE_HOP_PLANNER_MODEL,
            vector_searcher=self._vector,
            cypher_generator=self._cypher_gen,
            term_index=self._term_index,
            max_iter=ADAPTIVE_HOP_MAX_ITER,
            budget_ms=ADAPTIVE_HOP_BUDGET_MS,
        )

    async def _merge_title_search(
        self, question: str, graph_chunks: list[dict]
    ) -> tuple[list[dict], dict | None]:
        """RRF-merge the section-title retriever into the graph branch.

        Called AFTER the raw-score filter and score normalisation in both graph
        paths: a Lucene score is not a Pattern-A confidence, so running this
        earlier would let GRAPH_MIN_SCORE drop every title hit.

        Returns (chunks, sse_event_or_None) rather than yielding, so the two call
        sites keep their own control flow. Measured Recall@5 of A_kg_only:
        kg_crossspec_150 0.037 -> 0.167, wh_kg_v5_600 0.301 -> 0.512.
        """
        if ABL.title_search_off():
            return graph_chunks, None
        try:
            title_chunks = await asyncio.to_thread(
                self._title.search, question, TOP_K_TITLE
            )
        except Exception as exc:  # index missing on an old KG, Neo4j hiccup
            print(f"[title] search failed: {exc}", flush=True)
            return graph_chunks, None
        if not title_chunks:
            return graph_chunks, None
        merged = _title_rrf_merge(graph_chunks, title_chunks)
        event = {"stage": "retrieval_title", "data": {
            "input": {"top_k": TOP_K_TITLE},
            "output": {
                "count": len(title_chunks),
                "merged": len(merged),
                "top": _chunks_to_preview(title_chunks[:5]),
            },
            "count": len(title_chunks),
        }}
        return merged, event

    async def query(
        self,
        question: str,
        mode: str = "fixed",
        model: str = "qwen3:14b",
        think: bool = True,
        num_ctx: int | None = None,
        term_expansion_off: bool | None = None,
        rerank_off: bool | None = None,
        title_search_off: bool | None = None,
        kg_ablation: str | None = None,
        kg_evidence: str | None = None,
        provenance_on: bool | None = None,
        force_answer: bool = False,
    ) -> AsyncIterator[dict]:
        """
        Yield SSE event dicts: {stage, data}.
        Each non-token stage emits {input, output, ...flat fields} so the UI
        can show input/output of every tool when the user clicks Details.
        """
        # Pin ONE num_ctx for the whole request, right here at the entry point.
        # Ollama keys a loaded runner by (model, options): if any stage below sends a
        # different num_ctx it kills the running llama-server and reloads ~9GB of
        # weights mid-query. Callers that pass None (direct Python callers, scripts)
        # get the profile default / DEFAULT_NUM_CTX instead of Ollama's implicit 4096.
        num_ctx = resolve_num_ctx(model, num_ctx)

        # Ablation switches for THIS request. None inherits the process env, so a
        # server started with an ablation on keeps it unless the caller overrides.
        ABL.set_overrides(term_expansion_off=term_expansion_off,
                          rerank_off=rerank_off,
                          title_search_off=title_search_off,
                          kg_ablation=kg_ablation,
                          kg_evidence=kg_evidence,
                          provenance_on=provenance_on)

        # Step 1: combined intent + term extraction in ONE LLM call (format=json,
        # think=False). LLM returns intent + abbreviations + full_names + spec_refs;
        # TermFirstStrategy then HARD-VALIDATES each candidate against the live
        # KG TermIndex so hallucinations (e.g. "TELL") never reach the Cypher gen.
        # If the LLM fails (timeout, bad JSON), fall back to (regex intent +
        # deterministic TermIndex extraction) — still KG-backed, no hard-coded list.
        # num_ctx is passed here too: every Ollama call in one request must share
        # the same value or Ollama unloads/reloads the model between stages.
        intent_terms = await asyncio.to_thread(
            self._intent_clf.classify_with_terms, question, model, num_ctx
        )
        if intent_terms is not None:
            intent = intent_terms["intent"]
            terms = self._term_first.extract_with_llm_terms(
                question,
                intent_terms["abbreviations"],
                intent_terms["full_names"],
                intent_terms["spec_refs"],
            )
        else:
            intent = await asyncio.to_thread(
                self._intent_clf.classify, question, model, num_ctx
            )
            terms = self._term_first.extract_fallback(question)
        resolved = terms.get("resolved", {})
        # `ablations` is emitted so a recorded run can be audited after the fact
        # instead of trusting the config table that launched it. This repo has twice
        # measured configs that silently ran with the wrong environment; the trace
        # now carries what was actually in force for the request.
        yield {"stage": "intent", "data": {
            "input": {"question": question, "mode": mode},
            "output": {"intent": intent, "terms": terms, "ablations": ABL.active()},
            "intent": intent,
            "terms": terms,
            "ablations": ABL.active(),
        }}

        # Step 2 — RETRIEVAL: branch on mode.
        # primary_term is "" when nothing resolved against the KG (see
        # term_first._pick_primary) — don't seed with a blank string.
        seeds = terms["network_functions"] or ([terms["primary_term"]] if terms["primary_term"] else [])
        # Captured from hop_research_done — fed into per-gap rerank below so each
        # sub-question gets its own slot in the final top_k (avoids "compound query
        # bias" where the cross-encoder favours chunks that mention many topics).
        research_gaps: list[str] = []
        # Set when the graph branch returns nothing (Pattern C/D produced no rows).
        # Drives a rerank floor bump so the best vector hits still reach the answer
        # LLM instead of an empty context — see Step 3.
        graph_empty = False
        # Standardized-value concepts (Layer C) named verbatim in the question —
        # resolved once against the KG's Concept registry (cached, word-boundary).
        # Drives BOTH the Pattern-V block in the Cypher-gen prompt and the
        # value-anchor inside fetch_anchor_chunks. KG-bearing modes only, so the
        # Vector ablation stays free of graph assistance.
        value_concepts: list[str] = []
        if mode in ("fixed", "hybrid", "kg_only", "react_agent"):
            value_concepts = await asyncio.to_thread(
                detect_value_concepts, self._driver, question
            )
        if mode == "react_agent":
            candidates: list[dict] = []
            # Adaptive ReAct: planner LLM picks each tool (vector_search / cypher_query
            # / expand_term / inspect_chunk / finish) and decides when to stop.
            adaptive_iter = self._adaptive_hop.search_streaming(
                question=question,
                intent=intent,
                seeds=seeds,
                resolved_terms=resolved,
                prior_chunks=None,
                think=think,
                model=model,
                num_ctx=num_ctx,
            )
            async for ev in _async_iter(adaptive_iter):
                if ev.get("stage") == "hop_research_done":
                    research_gaps = (ev.get("data") or {}).get("gaps") or []
                if ev.get("stage") == "hop_finish":
                    candidates = (ev.get("data") or {}).pop("chunks", []) or []
                yield ev
            yield {"stage": "retrieval_adaptive", "data": {
                "input": {"seeds": seeds, "mode": "react_agent"},
                "output": {
                    "count": len(candidates),
                    "top": _chunks_to_preview(candidates),
                },
                "count": len(candidates),
                "seeds": seeds,
                "top": _chunks_to_preview(candidates),
            }}
        elif mode == "llm_only":
            # Ablation: NO retrieval — pure parametric. Answer LLM sees the question
            # only (empty context). Baseline for component-contribution analysis.
            candidates = []
            yield {"stage": "retrieval_vector", "data": {
                "input": {"mode": "llm_only"},
                "output": {"count": 0, "top": []},
                "count": 0, "top": [],
            }}
        elif mode == "vector_only":
            # Ablation: vector retrieval ONLY (no graph/KG branch). Isolates the
            # contribution of RAG-Vector vs. +KG.
            vec_results = await asyncio.to_thread(self._vector.search, question, top_k=TOP_K_VECTOR)
            yield {"stage": "retrieval_vector", "data": {
                "input": {"query": question, "top_k": TOP_K_VECTOR, "mode": "vector_only"},
                "output": {"count": len(vec_results), "top": _chunks_to_preview(vec_results)},
                "count": len(vec_results), "top": _chunks_to_preview(vec_results),
            }}
            candidates = vec_results
            graph_empty = True  # no graph → let the rerank floor keep vector hits
            # Stub retrieval_graph emitted AFTER vector (same relative order as
            # "fixed": vector step then graph step) so the SSE trail's step
            # skeleton is consistent across modes — only kg_only vs vector_only
            # differs in which of the two carries real data.
            yield {"stage": "retrieval_graph", "data": {
                "input": {"mode": "vector_only"},
                "output": {"count": 0, "top": []},
                "count": 0, "top": [],
            }}
        elif mode == "bm25":
            # Baseline: BM25 (sparse) retrieval ONLY — the Lucene/BM25 counterpart
            # of vector_only. No KG, no dense vectors. Falls through to the same
            # Step 3 rerank as vector_only so the comparison is apples-to-apples.
            bm25_results = await asyncio.to_thread(self._bm25.search, question, top_k=TOP_K_VECTOR)
            if _bm25_norm_on():
                bm25_results = _normalise_bm25_scores(bm25_results)
            yield {"stage": "retrieval_bm25", "data": {
                "input": {"query": question, "top_k": TOP_K_VECTOR, "mode": "bm25",
                          "score_norm": _bm25_norm_on()},
                "output": {"count": len(bm25_results), "top": _chunks_to_preview(bm25_results)},
                "count": len(bm25_results), "top": _chunks_to_preview(bm25_results),
            }}
            candidates = bm25_results
            graph_empty = True
            yield {"stage": "retrieval_graph", "data": {
                "input": {"mode": "bm25"},
                "output": {"count": 0, "top": []},
                "count": 0, "top": [],
            }}
        elif mode == "bm25_dense":
            # Baseline: sparse+dense hybrid (paper "Hybrid" row) — BM25 and dense
            # vector retrieval fused by RRF, NO KG branch. Isolates what sparse+dense
            # fusion buys over either alone, independent of the knowledge graph.
            bm25_results = await asyncio.to_thread(self._bm25.search, question, top_k=TOP_K_VECTOR)
            if _bm25_norm_on():
                bm25_results = _normalise_bm25_scores(bm25_results)
            yield {"stage": "retrieval_bm25", "data": {
                "input": {"query": question, "top_k": TOP_K_VECTOR, "mode": "bm25_dense",
                          "score_norm": _bm25_norm_on()},
                "output": {"count": len(bm25_results), "top": _chunks_to_preview(bm25_results)},
                "count": len(bm25_results), "top": _chunks_to_preview(bm25_results),
            }}
            vec_results = await asyncio.to_thread(self._vector.search, question, top_k=TOP_K_VECTOR)
            yield {"stage": "retrieval_vector", "data": {
                "input": {"query": question, "top_k": TOP_K_VECTOR, "mode": "bm25_dense"},
                "output": {"count": len(vec_results), "top": _chunks_to_preview(vec_results)},
                "count": len(vec_results), "top": _chunks_to_preview(vec_results),
            }}
            candidates = rrf_fusion(
                [bm25_results, vec_results],
                weights=[1.0, 1.0],
                top_k=TOP_K_VECTOR + TOP_K_GRAPH,
            )
            graph_empty = True
            yield {"stage": "retrieval_graph", "data": {
                "input": {"mode": "bm25_dense"},
                "output": {"count": 0, "top": []},
                "count": 0, "top": [],
            }}
        elif mode == "kg_only":
            # Ablation: graph/KG retrieval ONLY (no vector branch). Isolates the
            # contribution of KG vs. vector — symmetric with "vector_only" above.
            # Stub retrieval_vector emitted FIRST (before graph), same stage order
            # as "fixed" mode (vector step always precedes graph step in the SSE
            # trail) — keeps the pipeline step sequence consistent across modes
            # even though no real vector search runs here.
            yield {"stage": "retrieval_vector", "data": {
                "input": {"mode": "kg_only"},
                "output": {"count": 0, "top": []},
                "count": 0, "top": [],
            }}
            graph_chunks: list[dict] = []
            # KG-ablation "entity" level: relation-traversal (graph Cypher) is off,
            # so kg_only rests on the entity-node anchors added in the shared block
            # below. Higher levels run the graph branch as usual.
            if _kg_layer_on("relation"):
                graph_iter = self._graph.search_streaming(
                    question,
                    intent=intent,
                    term=terms["primary_term"],
                    resolved_terms=resolved,
                    top_k=TOP_K_GRAPH,
                    think=think,
                    vector_hints=None,
                    spec_refs=terms.get("spec_refs"),
                    value_concepts=value_concepts or None,
                    model=model,
                    num_ctx=num_ctx,
                )
                async for ev in _async_iter(graph_iter):
                    if ev.get("stage") == "retrieval_graph":
                        data = ev.get("data") or {}
                        graph_chunks = data.pop("_chunks", []) or []
                    yield ev
                graph_chunks = _filter_low_confidence_graph(graph_chunks)
            else:
                # KG-ablation "entity" level: no relation traversal — fall back to a
                # bare MENTIONS lookup over the question's resolved terms.
                graph_chunks = await asyncio.to_thread(
                    _bare_term_retrieval, self._driver, list(resolved.keys()), TOP_K_GRAPH
                )
                yield {"stage": "retrieval_graph", "data": {
                    "input": {"mode": "kg_only", "kg_ablation": "entity"},
                    "output": {"count": len(graph_chunks), "top": _chunks_to_preview(graph_chunks)},
                    "count": len(graph_chunks), "top": _chunks_to_preview(graph_chunks),
                }}
            # Same title merge as the fixed branch. kg_only has no vector branch to
            # fall back on, so a Cypher that returns nothing (26.7% of
            # kg_crossspec_150) or wrong-spec rows the reranker then drops (28.7%)
            # means an empty context and a refusal. Normalise first: this branch
            # skipped _normalise_graph_scores, and rrf_merge assumes [0,1].
            graph_chunks = _normalise_graph_scores(graph_chunks)
            # Mark chunks whose Steps put >=2 of the question's terms together. These
            # are exempt from the cross-encoder floor in fusion — a "which procedures
            # involve both" question shares no wording with the clause that answers
            # it, so the floor rejects the right chunk for a structural reason, not a
            # quality one. Marked here, AFTER Cypher, so the claim is checked against
            # the graph rather than inferred from whichever pattern the LLM wrote.
            await asyncio.to_thread(
                mark_step_actor_chunks, self._driver, graph_chunks, list(resolved))
            # NO title search in kg_only. It is a Lucene full-text branch over
            # section_title — a LEXICAL method — and mixing it in made this arm
            # something other than what its name says: measured on kg_req_100 it
            # fired on 100% of questions and was the ONLY source on 30%, and on the
            # 300-question set it lifted the requirement class from 0.130 to 0.300
            # while contributing nothing to the two classes the graph actually wins
            # (kg_value 0.785 pure vs 0.773 with it; kg_procedure 0.653 vs 0.655).
            # A row labelled "KG" must not carry a lexical retriever inside it. The
            # `fixed` branch keeps the merge — there it is one component of an openly
            # combined system, not a mislabelled single-method arm.
            candidates = graph_chunks
            graph_empty = len(graph_chunks) == 0
        else:
            # Fixed pipeline: deterministic vector + LLM-Cypher graph search,
            # merged via Reciprocal Rank Fusion. No adaptive loop.
            # Vector search (sentence-transformers encode + Neo4j query) is sync
            # blocking → to_thread so the event loop can tick between the intent
            # event and the retrieval_vector event.
            vec_results = await asyncio.to_thread(self._vector.search, question, top_k=TOP_K_VECTOR)
            yield {"stage": "retrieval_vector", "data": {
                "input": {"query": question, "top_k": TOP_K_VECTOR},
                "output": {
                    "count": len(vec_results),
                    "top": _chunks_to_preview(vec_results),
                },
                "count": len(vec_results),
                "top": _chunks_to_preview(vec_results),
            }}
            # Pass top-3 vector hits as anchors for the Cypher generator —
            # only title + spec_id, NOT content (keeps graph scoring independent
            # of vector scoring; mitigates Pattern C over-firing when vector
            # already found a real section anchor).
            vector_hints = [
                {"section": v.get("section", "?"), "spec_id": v.get("spec_id", "?")}
                for v in vec_results[:3]
            ]
            graph_chunks: list[dict] = []
            # Sync iterator (Ollama iter_lines) → wrap in _async_iter so the event
            # loop is free between each token. Without this wrap the cypher token
            # stream stays "silent" until the whole graph search finishes, then
            # flushes all at once.
            graph_iter = self._graph.search_streaming(
                question,
                intent=intent,
                term=terms["primary_term"],
                resolved_terms=resolved,
                # Retrieve WIDE (see TOP_K_GRAPH_WIDE): the in-Cypher LIMIT is a
                # blunt cut over constant-score Pattern F/G results, so we widen
                # it and let the cross-encoder below pick the real top-TOP_K_GRAPH.
                top_k=TOP_K_GRAPH_WIDE,
                think=think,
                vector_hints=vector_hints,
                spec_refs=terms.get("spec_refs"),
                value_concepts=value_concepts or None,
                # Single-model run: Cypher gen uses the SAME model as the answer
                # stage so Ollama keeps one set of weights resident the whole
                # query. Avoids 15-30s load/unload thrash per question.
                model=model,
                num_ctx=num_ctx,
            )
            async for ev in _async_iter(graph_iter):
                if ev.get("stage") == "retrieval_graph":
                    data = ev.get("data") or {}
                    graph_chunks = data.pop("_chunks", []) or []
                yield ev
            # Filter on the RAW score first: Pattern A's 0.6-tier noise gets cut by
            # GRAPH_MIN_SCORE=0.75 as before, while Pattern F/G counts (≥1) all pass.
            # THEN normalise to [0,1] so the rerank blend (_BLEND_BETA*upstream) and
            # the flat-score guard read a bounded scale — Pattern F/G emit raw counts
            # (n_steps=13, n_via_op=2) that would otherwise swamp the cross-encoder.
            graph_chunks = _filter_low_confidence_graph(graph_chunks)
            graph_chunks = _normalise_graph_scores(graph_chunks)
            # Mark chunks whose Steps put >=2 of the question's terms together. These
            # are exempt from the cross-encoder floor in fusion — a "which procedures
            # involve both" question shares no wording with the clause that answers
            # it, so the floor rejects the right chunk for a structural reason, not a
            # quality one. Marked here, AFTER Cypher, so the claim is checked against
            # the graph rather than inferred from whichever pattern the LLM wrote.
            await asyncio.to_thread(
                mark_step_actor_chunks, self._driver, graph_chunks, list(resolved))
            graph_chunks, _title_ev = await self._merge_title_search(question, graph_chunks)
            if _title_ev:
                yield _title_ev
            # Definition/functionality bias: demote TR studies, promote functional
            # sections so the answer cites the normative TS, not a study (see
            # _spec_type_bias — "SCP functionalities" case, 2026-07-12).
            if SPEC_BIAS_ON and intent in _SPEC_BIAS_INTENTS:
                graph_chunks = _spec_type_bias(graph_chunks, resolved)
            if PROC_GEN_BIAS_ON and intent in _PROC_BIAS_INTENTS:
                graph_chunks = _procedure_generality_bias(graph_chunks)
            # Chunks whose section names the procedure the question is about —
            # hard-protected from the cross-encoder cut below (see
            # _procedure_matched_ids). Fixes the 4-round "core registration chunk
            # never survives the 100→8 cut" root cause.
            proc_kws = _question_procedure_keywords(question)
            protected_ids = _procedure_matched_ids(graph_chunks, proc_kws)
            # Internal graph rerank (option 3): the wide Cypher result is ordered
            # by score alone (Pattern F/G's count) — that ranks structurally but not
            # by semantic relevance to the question. Run the cross-encoder over the
            # graph chunks and keep the top TOP_K_GRAPH BY RELEVANCE, so the chunks
            # that actually answer the question (incl. gold) reach RRF instead of an
            # arbitrary Neo4j-order slice. Skip when few chunks (cut is a no-op).
            if len(graph_chunks) > TOP_K_GRAPH:
                by_id = {c.get("chunk_id"): c for c in graph_chunks}
                reranked_graph = await asyncio.to_thread(
                    rerank, question, graph_chunks, top_k=TOP_K_GRAPH,
                    resolved_terms=resolved, min_keep=TOP_K_GRAPH,
                )
                # Union in any procedure-matched chunk the rerank dropped, capped
                # at PROC_PROTECT_MAX so a keyword like "registration" that matches
                # many chunks can't flood the graph pool. Ranked by graph score so
                # the strongest structural matches (most AMF/UDM mentions+ops) win.
                kept = {c.get("chunk_id") for c in reranked_graph}
                dropped_protected = sorted(
                    (by_id[i] for i in protected_ids if i in by_id and i not in kept),
                    key=lambda c: c.get("score") or 0, reverse=True,
                )[:PROC_PROTECT_MAX]
                graph_chunks = reranked_graph + dropped_protected
            graph_empty = len(graph_chunks) == 0
            # RRF pool = every candidate from both branches (vector + graph), so
            # no chunk is dropped before the cross-encoder rerank — the rerank is
            # the real quality gate and should see all candidates. Previously the
            # pool was capped at TOP_K_FINAL*2=16 < 10+8=18, so with a strong
            # graph weight up to 2 vector hits (which could hold the gold) were
            # evicted before rerank ever saw them (BAO_CAO_5_MODE.md, G-005).
            # BM25_BRANCH adds the sparse branch to the proposed system's fusion.
            # It exists because fixing the BM25 query tokenizer (2026-08-17) made
            # sparse the strongest single retriever, and this pipeline had never
            # included it — so "does the KG add anything to a STRONG baseline?"
            # was unanswerable: the system was only ever fused with a weaker one.
            branches, weights = [vec_results], [1.0]
            if BM25_BRANCH:
                bm25_extra = await asyncio.to_thread(
                    self._bm25.search, question, top_k=TOP_K_VECTOR)
                branches.append(bm25_extra)
                weights.append(1.0)
                yield {"stage": "retrieval_bm25", "data": {"count": len(bm25_extra)}}
            branches.append(graph_chunks)
            weights.append(_graph_rrf_weight(graph_chunks))
            candidates = rrf_fusion(
                branches, weights=weights,
                top_k=TOP_K_VECTOR * len(branches) + TOP_K_GRAPH,
            )

        # Definition/functionality bias on the fused pool: also tilt the vector
        # branch (RRF ranks by position, so the graph-side bias above doesn't reach
        # vector hits — apply it here so both branches feed the final rerank with
        # TR studies demoted and functional sections promoted).
        if SPEC_BIAS_ON and intent in _SPEC_BIAS_INTENTS:
            candidates = _spec_type_bias(candidates, resolved)
        if PROC_GEN_BIAS_ON and intent in _PROC_BIAS_INTENTS:
            candidates = _procedure_generality_bias(candidates)

        # G4 anchor retrieval: pull the question's procedure ROOT clause straight
        # from the KG (e.g. registration → ts_23_502_4.2.2.2.2 "General
        # Registration") and add it to the candidate pool with parent_of_anchor=True
        # so the Part E rerank bias boosts it. This is the fix for the clause that
        # never reached context across five rounds — it's structurally low-scoring
        # (definition-typed, no Step) so neither vector nor FG surfaces it, but it
        # holds the canonical UECM_Registration → SDM_Get → SDM_Subscribe flow.
        # Injected for procedure/interaction questions only (gate on intent, not
        # just a keyword match — "what is the Registration Management state machine"
        # contains "registration" but wants a definition, not the procedure root
        # clause). Deduped by chunk_id.
        # 2026-08-08 rework: (a) anchors now fire for every KG-bearing mode and
        # every intent — the verbatim anchors (quoted phrase, SBI op name, TS
        # clause citation) are precision signals independent of intent, and
        # fetch_anchor_chunks gates the procedure-ROOT part on intent itself.
        # (b) vector_only no longer receives anchors: anchors are KG lookups, and
        # the thesis proposal's controlled pair (Vector vs Hybrid) must differ in exactly
        # the KG component. (c) soft anchors (per-op/service matches that may sit
        # beside, not on, the answer) only join the candidate pool for the
        # cross-encoder to judge — they are never forced to the context front.
        anchors: list[dict] = []
        # Anchors belong to the "relation" KG-ablation layer (they exploit
        # DESCRIBES_OPERATION / PROVIDED_BY / section-title lookups) — off at the
        # bare "entity" level so its recall stays genuinely low. No-op when the
        # ablation gate is off (_kg_layer_on always True then).
        if ANCHOR_ON and mode in ("fixed", "hybrid", "kg_only", "react_agent") and _kg_layer_on("relation"):
            all_anchors = await asyncio.to_thread(
                fetch_anchor_chunks, self._driver, question, intent
            )
            # Hard anchors only — Step 3a0 must never front-pull a soft anchor.
            anchors = [a for a in all_anchors if not a.get("soft_anchor")]
            if all_anchors:
                have = {c.get("chunk_id") for c in candidates}
                candidates = candidates + [
                    a for a in all_anchors if a.get("chunk_id") not in have
                ]

        # Reference-chain hop expansion (see _HOP_INTENTS above): follow
        # REFERENCES_CHUNK 1-2 hops from the current best candidates, to recover
        # cross-spec cited chunks no existing Cypher pattern reaches (measured
        # in de_cuong/PhuLuc_B_5ca_vector_vs_fixed.md). Seeds are the top-ranked
        # RRF candidates PLUS any G4 anchor (an anchor is often the procedure
        # ROOT clause, exactly the kind of chunk that cites the cross-spec
        # parameter/format detail a sub-question asks about) — anchors are
        # unioned in explicitly since they may sit past HOP_SEED_TOP_N in
        # `candidates` (appended after the RRF-ranked slice).
        # KG-ablation "xref" layer: reference-hop is normally fixed-only, but when a
        # KG_ABLATION level is active it also runs in kg_only so the cross-reference
        # layer has an effect there (gated off below the xref level).
        _hop_mode_ok = mode == "fixed" or (ABL.kg_ablation() in ABL.KG_ABL_ORDER
                                           and mode == "kg_only")
        if (HOP_EXPANSION_ON and _hop_mode_ok and _kg_layer_on("xref")
                and intent in _HOP_INTENTS):
            seed_ids = [c.get("chunk_id") for c in candidates[:HOP_SEED_TOP_N] if c.get("chunk_id")]
            anchor_ids = [a.get("chunk_id") for a in anchors if a.get("chunk_id")]
            seed_ids = list(dict.fromkeys(seed_ids + anchor_ids))
            hop_chunks = await asyncio.to_thread(
                expand_reference_chain, self._driver, seed_ids,
                max_targets=HOP_MAX_TARGETS,
            ) if seed_ids else []
            have = {c.get("chunk_id") for c in candidates}
            new_hop_chunks = [h for h in hop_chunks if h.get("chunk_id") not in have]
            preview = [{
                "spec_id": c.get("spec_id", "?"),
                "section": c.get("section", "?"),
                "hops": c.get("hops"),
                "via_seed": c.get("hop_via_seed"),
            } for c in new_hop_chunks[:3]]
            yield {"stage": "retrieval_hop", "data": {
                "input": {"seeds": seed_ids, "intent": intent},
                "output": {"count": len(new_hop_chunks), "top": preview},
                "count": len(new_hop_chunks),
                "top": preview,
                "seeds": seed_ids,
            }}
            if new_hop_chunks:
                candidates = candidates + new_hop_chunks

        # Step 3: cross-encoder rerank picks the final top_k for the answer prompt.
        # When research_gaps is non-empty (react_agent mode), use per-gap reranking
        # so each sub-question gets its own slot — this fixes the compound-query
        # bias where chunks that vaguely match many topics outrank chunks that
        # squarely answer one specific gap.
        # Rerank uses a cross-encoder (sentence-transformers, sync) — to_thread so
        # the event loop can tick before yielding the rerank event.
        if mode == "hybrid" or ABL.rerank_off():
            # Ablation: RRF fusion WITHOUT the final cross-encoder rerank — isolates
            # the reranker's own contribution when compared against "fixed" (same
            # retrieval, rerank included). Use the RRF-fused order directly.
            # RERANK_OFF extends the same ablation to any other mode.
            reranked = candidates[:TOP_K_FINAL]
        elif mode == "react_agent" and len(research_gaps) > 1:
            reranked = await asyncio.to_thread(
                rerank_per_gap,
                question=question,
                gaps=research_gaps,
                chunks=candidates,
                total_top_k=TOP_K_FINAL,
                resolved_terms=resolved,
            )
        else:
            # When the graph branch was empty, ask rerank to keep a floor of
            # chunks so the answer LLM doesn't run on empty context. The floor is
            # logit-gated inside _rerank_with_dedup: only chunks the cross-encoder
            # isn't confident are irrelevant get revived — so in-domain graph_count=0
            # questions keep their hits (+5.42pp) while out-of-domain (Research)
            # questions get empty context instead of off-topic 3GPP noise (recovers −6pp).
            reranked = await asyncio.to_thread(
                rerank, question, candidates, top_k=TOP_K_FINAL,
                resolved_terms=resolved,
                min_keep=RERANK_GRAPH_EMPTY_MIN_KEEP if graph_empty else None,
            )

        # Step 3a-vec: RESERVED SLOTS for the top vector hits. Measured on
        # wh_kg_v3_100 (2026-07-12): for procedure/interaction questions the graph
        # branch floods the RRF pool with ~100 same-structural-score procedure
        # chunks (many procedures in one spec share the asked NF set) that the
        # cross-encoder can't tell apart from the real gold, so they evict the
        # vector branch's gold — which sat at vector rank 1-2 (F-003 rank 2, G-007
        # rank 1). We guarantee the top-N vector hits reach the answer context.
        # GATED to procedure/interaction intents OR interface-identifier questions.
        # For a definition question naming a message/parameter (Pattern M/P), the
        # graph branch IS the gold (DESCRIBES_MESSAGE / DEFINED_IN_TABLE) and the top
        # vector hits are noise — reserving vector slots there evicts the M/P gold
        # (measured M 0.64→0.38, P 0.54→0.34). But an interface / reference-point
        # question (N26, N6, S1…) uses Pattern B (section_title regex) which returns
        # a FLAT-score flood of every chunk whose title contains the identifier —
        # that evicts the vector gold too (CAP-T4-007 "N26 interface": gold sat at
        # vector rank 5, evicted by 40 N26-title chunks). Detect it via the flat
        # graph score + an interface identifier in the question, and protect vector.
        _iface_q = bool(_INTERFACE_ID_RE.search(question))
        if (
            (intent in _PROC_BIAS_INTENTS or _iface_q)
            # bm25 / bm25_dense have no graph branch, so there is no flood to
            # protect the vector hits from — the reservation would only bias the
            # sparse/dense baseline toward its dense side. Exclude them.
            and mode not in ("react_agent", "kg_only", "llm_only", "hybrid", "bm25", "bm25_dense")
            and reranked is not None
        ):
            try:
                vec_top = vec_results[:VECTOR_RESERVED_SLOTS]
            except NameError:
                vec_top = []
            kept = {c.get("chunk_id") for c in reranked}
            missing_vec = [v for v in vec_top if v.get("chunk_id") not in kept]
            if missing_vec:
                # Prepend the missing vector hits, trim from the tail to keep top_k.
                keep_tail = max(0, TOP_K_FINAL - len(missing_vec))
                reranked = missing_vec + reranked[:keep_tail]

        # Step 3a0: RESERVED SLOT for the G4 anchor (procedure root clause). The
        # anchor is identified structurally (section_title == "General <proc>" in a
        # normative TS), so it does NOT need the cross-encoder to vouch for it — and
        # it must not: the root Registration clause (ts_23_502_4.2.2.2.2) is an 88 KB
        # chunk whose first 512 tokens are the RAN steps, so the cross-encoder scores
        # it −2.2 (it never "sees" the AMF↔UDM steps buried deep in the body) and
        # drops it every time — the exact reason it never reached context in five
        # rounds. We force it into the pool; _fit_chunk_content below sub-chunks the
        # 88 KB down to the AMF↔UDM-relevant segments by relevance, so the size isn't
        # a problem once it's IN. Dedup against what rerank already kept.
        if anchors and reranked is not None:
            # The anchor is a verbatim section-title / procedure-root match, so it
            # is trusted above the cross-encoder verdict — move it to the FRONT
            # whether the rerank dropped it OR merely buried it below top-5 (the
            # AF-QoS case: gold ts_23_502_4.15.6.6 reached the pool but sat at rank
            # >5 under FG-flood siblings). Pull every anchor to the lead, keep the
            # rest of the reranked order, trim the tail to top_k.
            anchor_ids = {a.get("chunk_id") for a in anchors}
            by_id = {c.get("chunk_id"): c for c in reranked}
            # Anchor dicts we already reranked (keep their richer fields) else the
            # raw anchor chunk.
            lead = [by_id.get(a.get("chunk_id"), a) for a in anchors]
            rest = [c for c in reranked if c.get("chunk_id") not in anchor_ids]
            reranked = lead + rest[: max(0, TOP_K_FINAL - len(lead))]

        # Step 3a: float procedure-matched chunks to the FRONT of the context.
        # The reviewer's 4-round finding: even when the core-procedure chunk
        # (ts_23_502_4.13.3.1, SMS-over-NAS Registration) reaches the context, the
        # LLM reads chunks in order and anchors on whatever leads — so a chunk whose
        # section actually NAMES the asked-about procedure must lead, not sit at
        # rank 6 under reachability/interworking chunks. Stable partial reorder:
        # keeps relative order within each group. (Anchors already lead from 3a0 and
        # also name the procedure, so they stay in front.)
        proc_kws_final = _question_procedure_keywords(question) if PROC_FRONT_ON else []
        if proc_kws_final and reranked:
            def _names_procedure(c: dict) -> bool:
                title = (c.get("section") or c.get("section_title") or "").lower()
                return any(kw in title for kw in proc_kws_final)
            # Hard anchors stay in the lead unconditionally — an op-defining
            # clause is titled "ConfigCreate", never the procedure keyword, and
            # this reorder used to push keyword-titled noise above it (2026-08-08).
            anchor_ids_3a = {a.get("chunk_id") for a in anchors}
            lead = [c for c in reranked if c.get("chunk_id") in anchor_ids_3a]
            tail = [c for c in reranked if c.get("chunk_id") not in anchor_ids_3a]
            front = [c for c in tail if _names_procedure(c)]
            rest = [c for c in tail if not _names_procedure(c)]
            reranked = lead + front + rest

        # Step 3b: neighbor expansion — top up with PARENT_SECTION siblings of
        # the reranked chunks (see _expand_neighbors docstring). Additive only:
        # appends up to NEIGHBOR_MAX_EXTRA chunks, never displaces a chunk the
        # reranker already picked. No-op for llm_only (reranked is empty there).
        # KG-ablation "hierarchy" layer gates the PARENT_SECTION traversal.
        if reranked and _kg_layer_on("hierarchy"):
            existing_ids = {c.get("chunk_id") for c in reranked if c.get("chunk_id")}
            neighbors = await asyncio.to_thread(
                _expand_neighbors, self._driver, list(existing_ids), existing_ids, NEIGHBOR_MAX_EXTRA,
            )
            reranked = reranked + neighbors
        yield {"stage": "rerank", "data": {
            "input": {
                "query": question,
                "candidate_count": len(candidates),
                "top_k_final": TOP_K_FINAL,
                "gap_count": len(research_gaps),
                "rerank_mode": "per_gap" if (mode == "react_agent" and len(research_gaps) > 1) else "single",
            },
            "output": {
                "count": len(reranked),
                "top": _chunks_to_preview(reranked, score_key="final_score"),
            },
            "count": len(reranked),
            "top": _chunks_to_preview(reranked, score_key="final_score"),
            # Full ordered chunk_id list (ids only, no content) — lets the
            # retrieval benchmark measure Recall@k from the rerank event without
            # waiting for the (slow) answer LLM. Ignored by the UI trail.
            "chunk_ids": [c.get("chunk_id") for c in reranked if c.get("chunk_id")],
        }}

        # Step 3c: nothing retrieved → refuse here instead of asking the LLM to
        # answer from an empty context. Measured on the KG-ablation configs (graph
        # branch only, ~20% of questions retrieve nothing): with no context the
        # model fills the "sources" section by enumerating invented spec numbers —
        # one answer looped `**Full title**: Not provided in the context` 275 times,
        # 41,857 chars, 201 s, because the answer stage has no num_predict cap and
        # only stops near the context ceiling.
        # Skipped for llm_only, whose whole point is answering with no context.
        if ABSTAIN_ON_EMPTY and mode != "llm_only" and not reranked and not force_answer:
            # English, like every other answer this system produces: the corpus, the
            # prompts and the questions are English, and a Vietnamese-only refusal is
            # both inconsistent for the reader and a measurement hazard — an
            # English-language abstain detector silently miscounted it 6/10 instead of
            # 10/10 during analysis. bench_metrics.ABSTAIN_SENTINELS matches BOTH this
            # string and the former Vietnamese one, so historical runs still classify.
            msg = ("No relevant passage was found in the 3GPP corpus to answer "
                   "this question.")
            yield {"stage": "answer_start", "data": {
                "input": {"model": model, "prompt_chars": 0, "prompt": ""},
                "abstained": "no_retrieval",
            }}
            yield {"stage": "answer", "data": msg}
            yield {"stage": "sources", "data": []}
            return

        # Step 4: build prompt + stream LLM answer. llm_only uses a parametric
        # prompt (no context, no grounding rules) so the model answers from its
        # own internal knowledge.
        used_chunk_ids: list[str] | None = None   # None = "not budget-filtered"
        if mode == "llm_only" or (force_answer and not reranked):
            # force_answer with nothing retrieved: answer parametrically rather than
            # hand the grounded template an empty context.
            context = ""
            prompt = build_llm_only_prompt(question)
        elif ABL.kg_evidence() == "triples":
            # Study-Ablation "KG-only (KG triples)": feed serialized KG triples of
            # the retrieved chunks' neighbourhood instead of chunk prose.
            reranked_ids = [c.get("chunk_id") for c in reranked if c.get("chunk_id")]
            context = await asyncio.to_thread(
                serialize_triples, self._driver, reranked_ids
            )
            # This branch used to fall through WITHOUT assigning `prompt`, so the
            # generator died on NameError at `len(prompt)` below — the client saw a
            # stream that simply ended, with no error event. Measured: 482/600 empty
            # answers in the study_triples config, which made its whole row read 0.
            prompt = build_prompt(intent, context, question, force_answer=force_answer)
        else:
            # Step-scoped evidence (experiment, STEP_EVIDENCE=1). Runs AFTER rerank so
            # the retrieved set — and therefore Recall@5 — is identical whether it is
            # on or off; only what those chunks CONTRIBUTE to the prompt changes. That
            # isolation is the point: the experiment asks whether answer quality holds
            # when a 3,930-char clause is replaced by the ~320-char step that answers.
            if STEP_EVIDENCE:
                n_scoped = await asyncio.to_thread(
                    attach_step_evidence, self._driver, reranked, question, list(resolved)
                )
                if n_scoped:
                    saved = sum((c["step_evidence_stats"]["chars_full"]
                                 - c["step_evidence_stats"]["chars_scoped"])
                                for c in reranked if c.get("step_evidence_stats"))
                    yield {"stage": "step_evidence", "data": {
                        "input": {"resolved": list(resolved)},
                        "output": {"scoped": n_scoped, "chars_saved": saved},
                        "count": n_scoped,
                    }}
            # Sub-chunk long sections by relevance to the question (see
            # _fit_chunk_content) — runs the cross-encoder, so to_thread it.
            context, used_chunk_ids = await asyncio.to_thread(
                _chunks_to_context, reranked, question, 20000, 4000,
                ABL.provenance_on() and ABL.kg_layer_on("provenance")
            )
            # Out-of-scope / fabricated-entity check (pipeline.scope_check): flags
            # named entities the question is built around that the KG doesn't
            # contain, so the answer prompt can abstain instead of hallucinating.
            warns = await asyncio.to_thread(
                scope_warnings, question, self._driver, resolved
            )
            prompt = build_prompt(intent, context, question, scope_warnings=warns,
                                  force_answer=force_answer)

        yield {"stage": "answer_start", "data": {
            "input": {
                "model": model,
                "prompt_chars": len(prompt),
                "prompt": prompt,
            },
        }}

        # Stream thinking + response tokens. Reasoning models (deepseek-r1, qwen3)
        # emit thinking first when think=True; otherwise (or for non-reasoning models)
        # we go straight to the response phase.
        # Sync iter_lines from Ollama → wrap in _async_iter so each token flushes
        # immediately (instead of seeing nothing until the whole answer finishes).
        answer_tokens: list[str] = []
        answer_iter = self._llm.generate_stream_full(prompt, model=model, think=think, num_ctx=num_ctx)
        async for ev in _split_inline_think(_async_iter(answer_iter)):
            if ev["kind"] == "thinking":
                yield {"stage": "thinking", "data": ev["token"]}
            else:
                answer_tokens.append(ev["token"])
                yield {"stage": "answer", "data": ev["token"]}

        # Step 5: sources summary. `picked_for_gap` is set by rerank_per_gap so
        # the UI can show which sub-question each chunk was selected for.
        # When the whole answer is an uncited "Context does not cover ..."
        # (grounding rule #3), the reranked chunks weren't actually used to
        # ground anything — don't show them as if they were.
        full_answer = _strip_inline_think("".join(answer_tokens))
        if _answer_has_no_coverage(full_answer):
            sources = []
        else:
            # Only the chunks that actually reached the prompt. `reranked` is what
            # survived reranking; _chunks_to_context then drops whatever does not fit
            # the character budget, and those were never evidence for the answer.
            in_prompt = reranked if used_chunk_ids is None else [
                c for c in reranked if c.get("chunk_id") in set(used_chunk_ids)
            ]
            sources = [
                {
                    "spec_id": c.get("spec_id", "?"),
                    "section": c.get("section", "?"),
                    "chunk_id": c.get("chunk_id"),
                    "content": c.get("content", ""),
                    # Surface both the blended final_score (sort key) and the raw
                    # cross-encoder logit (for debugging) so trail UI can show either.
                    "score": c.get("final_score"),
                    "rerank_score": c.get("rerank_score"),
                    "picked_for_gap": c.get("picked_for_gap"),
                }
                for c in in_prompt
            ]
        yield {"stage": "sources", "data": sources}
