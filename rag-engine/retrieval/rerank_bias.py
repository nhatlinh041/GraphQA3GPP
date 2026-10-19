"""
Feature-based rerank bias — implements goal.md Part E (G1–G3) as an additive
term in the cross-encoder blend, so structural signals the cross-encoder can't
read (chunk_type, section_title shape, TR-vs-TS, procedure-context match) tilt
the final ranking directly instead of only through the upstream graph score.

The cross-encoder logit (α=0.6 of the blend) is the dominant term and is exactly
what the goal's five-round case study found to be *reversed*: it put a TR
"Impacts on services" bullet chunk on top and a real procedure-flow chunk at the
bottom. Multiplying the upstream score (β=0.3) — as the earlier orchestrator
biases did — can't override a large logit gap. This module instead adds a
bounded `bias ∈ [-1, +1]` as its own blend term (δ), applied inside
_rerank_with_dedup, so a mislabelled ranking can actually be corrected.

`compute_bias(chunk, question, resolved_terms)` returns a float in roughly
[-1, +1]; the blend adds `_BLEND_DELTA * bias`. Positive = promote (real
procedure flow, matching context, service-operation bridge); negative = demote
(TR study, "Impacts on…" bullet list, off-topic context, generic state machine).

All signals are cheap: string tests on section_title / spec_id / chunk_type plus
booleans the orchestrator precomputes on the chunk dict (`has_step_all_actors`,
`op_prefix_match`, `parent_of_anchor`, `via_reference_chain`). Nothing here
calls the KG or an LLM.
"""
from __future__ import annotations

import re
from typing import Optional

# ── Signal weights (goal.md Part E "strong / medium / light penalty", "strong / medium boost").
# Bounded so the summed bias stays roughly in [-1, +1]; the blend's _BLEND_DELTA
# scales the whole thing. Env-overridable for tuning.
import os

W_TR_PENALTY = float(os.getenv("BIAS_TR_PENALTY", "-0.6"))          # G2: Technical Report
W_IMPACTS_PENALTY = float(os.getenv("BIAS_IMPACTS_PENALTY", "-0.7"))  # G1: "Impacts on services/entities"
W_STATE_MACHINE_PENALTY = float(os.getenv("BIAS_STATEMACHINE_PENALTY", "-0.25"))  # G1: generic General/definition, no flow
W_STEP_BOTH_ACTORS_BOOST = float(os.getenv("BIAS_STEP_BOTH_BOOST", "0.7"))  # G1/G3: a Step involves BOTH asked entities
# Per EXTRA step beyond the first in which all asked entities meet, capped. One shared
# step can be incidental; several means the clause is ABOUT that interaction. Measured
# on kg_procedure: correct clauses average 2.26 such steps, other retrieved chunks 0.32,
# and 84% of the others have none at all — so the count separates far better than the
# boolean. Small per-step value on purpose: this orders chunks that already earned the
# boolean boost, it must not let a many-step chunk outrank a differently-justified one.
W_STEP_ACTOR_COUNT_BOOST = float(os.getenv("BIAS_STEP_COUNT_BOOST", "0.12"))
W_STEP_ACTOR_COUNT_CAP = float(os.getenv("BIAS_STEP_COUNT_CAP", "0.36"))
W_OP_PREFIX_BOOST = float(os.getenv("BIAS_OP_PREFIX_BOOST", "0.4"))  # G1: describes an operation of one entity
W_ANCHOR_PARENT_BOOST = float(os.getenv("BIAS_ANCHOR_PARENT_BOOST", "0.7"))  # G4: same section tree as the procedure's root clause
W_PROC_CONTEXT_BOOST = float(os.getenv("BIAS_PROC_CONTEXT_BOOST", "0.5"))  # G3: section names the asked-about procedure
W_OFFTOPIC_CONTEXT_PENALTY = float(os.getenv("BIAS_OFFTOPIC_PENALTY", "-0.3"))  # G3: contains entities but in a different procedure
# Reference-chain hop expansion (retrieval/reference_hop.py). Weaker prior than
# G4 anchor (0.7, exact section-title match) — a REFERENCES_CHUNK edge only
# confirms "some already-relevant chunk cites this one", not that it directly
# answers the question. Still needs to be a real boost (not just the canonical
# score=1.0 floor bypass in fusion.py) because these chunks are BY DEFINITION
# the ones weak on query<->chunk semantic similarity, which is most of what the
# cross-encoder logit measures.
W_REFERENCE_CHAIN_BOOST = float(os.getenv("BIAS_REFERENCE_CHAIN_BOOST", "0.5"))

# TR (Technical Report) spec-id shape: 2nd number series ≥ 700 (ts_23_7xx-yy).
_TR_SPEC_RE = re.compile(r"^ts_\d{2}_(\d{3})")

# "Impacts on services/entities and interfaces" — a TR bullet-list section that
# enumerates touched NFs without describing a flow. 848 such chunks, 92% from TRs.
_IMPACTS_RE = re.compile(r"impacts?\s+on\s+(services|existing|entities|the)", re.IGNORECASE)

# Procedure keywords: when the question is about procedure P, a chunk whose
# section_title names P is on-context (G3); one that names a DIFFERENT procedure
# while still carrying both entities is off-context. Kept aligned with the
# orchestrator's _PROCEDURE_KEYWORDS.
_PROCEDURE_KEYWORDS = (
    "registration", "deregistration", "authentication", "handover",
    "subscription", "selection", "establishment", "modification", "release",
    "reallocation", "reachability", "configuration update", "service request",
    "paging", "mobility", "provisioning", "activation", "deactivation",
)

# Generic conceptual sections that describe a state machine / definition rather
# than a flow — demoted mildly (G1) unless they also carry a real Step signal.
_GENERIC_TITLE_KW = ("general", "overview", "introduction", "definition", "principles")


def is_tr_spec(spec_id: str) -> bool:
    m = _TR_SPEC_RE.match(spec_id or "")
    return bool(m) and int(m.group(1)) >= 700


def question_procedures(question: str) -> list[str]:
    """Procedure keywords the question is about (drives G3 context match)."""
    q = (question or "").lower()
    return [kw for kw in _PROCEDURE_KEYWORDS if kw in q]


def _question_names_solution(question: str) -> bool:
    """True when the question explicitly asks about a TR solution/key issue/study
    — in that case a TR is on-topic and must NOT be penalised (G2 exception)."""
    q = (question or "").lower()
    return any(kw in q for kw in ("solution", "key issue", "study", "tr 23", "technical report"))


def compute_bias(
    chunk: dict,
    question: str,
    resolved_terms: Optional[dict] = None,
) -> float:
    """Additive rerank bias in roughly [-1, +1] implementing goal.md Part E.
    Reads section_title / spec_id / chunk_type plus optional booleans the
    orchestrator precomputes:
      chunk['has_step_all_actors']  — a Step INVOLVES every asked entity (G1/G3)
      chunk['op_prefix_match']      — DESCRIBES_OPERATION with a matching NF prefix (G1)
      chunk['parent_of_anchor']     — same PARENT_SECTION tree as the procedure root clause (G4)
      chunk['via_reference_chain'] — reached via REFERENCES_CHUNK hop from a trusted seed chunk
    Missing booleans default False (bias still works from the string signals)."""
    title = (chunk.get("section") or chunk.get("section_title") or "").strip()
    title_l = title.lower()
    spec_id = chunk.get("spec_id") or ""
    ctype = (chunk.get("chunk_type") or "").lower()
    bias = 0.0

    # ── G2: Technical Report penalty (unless the question is about that study).
    if is_tr_spec(spec_id) and not _question_names_solution(question):
        bias += W_TR_PENALTY

    # ── G1: "Impacts on services/entities" bullet-list penalty.
    if _IMPACTS_RE.search(title_l):
        bias += W_IMPACTS_PENALTY

    # ── G1/G3: a Step involves BOTH asked entities — the strongest "real flow"
    # signal (orchestrator sets this from HAS_STEP→INVOLVES).
    if chunk.get("has_step_all_actors"):
        bias += W_STEP_BOTH_ACTORS_BOOST
        # Graded on how many steps put them together — see W_STEP_ACTOR_COUNT_BOOST.
        extra = max(0, int(chunk.get("n_step_actor_hits") or 1) - 1)
        bias += min(W_STEP_ACTOR_COUNT_CAP, extra * W_STEP_ACTOR_COUNT_BOOST)
    # ── G1: describes a service operation whose NF prefix is one of the entities.
    if chunk.get("op_prefix_match"):
        bias += W_OP_PREFIX_BOOST
    # ── G4: same section tree as the procedure's root clause (anchor).
    if chunk.get("parent_of_anchor"):
        bias += W_ANCHOR_PARENT_BOOST
    # ── Reference-chain hop: reached via an authored REFERENCES_CHUNK citation
    # from a chunk retrieval already trusted (retrieval/reference_hop.py).
    if chunk.get("via_reference_chain"):
        bias += W_REFERENCE_CHAIN_BOOST

    # ── G3: procedure-context match / mismatch. Only fires when the question
    # names a procedure. The OFFTOPIC PENALTY half is skipped for reference-
    # chain hop chunks: that signal means "does this chunk's OWN section
    # belong to the asked procedure or a different one" — meaningless for a
    # chunk included because chunk A cites it, not because it independently
    # mentions the asked procedure/entities. Measured: a hop chunk whose title
    # happened to contain an UNRELATED procedure keyword (title "Paging Policy
    # Differentiation" containing "paging", question about "reachability") ate
    # the full G3 offtopic penalty (-0.3) despite being exactly the cited
    # answer chunk. The BOOST half is NOT skipped — a hop chunk whose title
    # DOES name the asked procedure is a genuinely stronger signal, not a
    # meaningless one, and q_procs itself is computed unconditionally so the
    # state-machine exemption below (which also reads q_procs) stays correct
    # instead of losing its "on-topic" exemption for hop chunks too.
    q_procs = question_procedures(question)
    if q_procs:
        names_asked_proc = any(p in title_l for p in q_procs)
        names_other_proc = any(
            kw in title_l for kw in _PROCEDURE_KEYWORDS if kw not in q_procs
        )
        if names_asked_proc:
            bias += W_PROC_CONTEXT_BOOST
        elif names_other_proc and not chunk.get("via_reference_chain"):
            # Carries entities but sits in a different procedure's section.
            bias += W_OFFTOPIC_CONTEXT_PENALTY

    # ── G1: generic conceptual section with no flow signal — mild penalty, but
    # never below a chunk that has a real step/operation signal (guarded above by
    # the boosts already added).
    if (
        not chunk.get("has_step_all_actors")
        and not chunk.get("op_prefix_match")
        and (ctype in ("definition",) or any(k == title_l for k in _GENERIC_TITLE_KW))
        and not any(p in title_l for p in q_procs)
    ):
        bias += W_STATE_MACHINE_PENALTY

    # Clamp so one runaway chunk can't dominate the blend.
    return max(-1.0, min(1.0, bias))
