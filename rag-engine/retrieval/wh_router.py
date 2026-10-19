"""
WH-template router (2026-07-12) — two-tier pattern selection.

Replaces the flat 10-pattern selection prompt with a structured router:

  Tier 1 (deterministic, no LLM): classify the question into ONE of 5 WH-types
          from the already-computed `intent`. English 3GPP questions don't
          reliably start with a WH-word ("Describe the SCP role" is a WHAT), so
          we route on the semantic intent, not the surface word.

  Tier 2 (deterministic + a tiny LLM slot-extraction only where a WH-type needs
          it): each WH-type owns a small decision tree over #terms / spec_refs
          that picks the concrete Cypher pattern (FUNC/A1/A2/F/G/FG/B/M/P/E/...).

The five WH-types and their sub-pattern trees:

  WHAT_DEF  (definition / functionality / capability / list)
      1 term  → FUNC   (term → defining spec → the term's OWN functional section)
      ≥2 term → A2
      0 term  → M / P / B / C  (named message / parameter / interface / sentinel)

  HOW_PROC  (procedure / how_does / call-flow / mechanism)
      ≥2 term → FG   (step co-occurrence ∪ service-operation bridge)
      1 term  → A1   (single-actor procedure lookup)
      0 term  → M / B / C

  WHERE_LOC (location / which-spec / interface / reference-point)
      spec_ref named          → E   (cross-spec citation)
      interface_id present     → B   (section_title regex)
      else                     → A1 / A2 / C

  WHO_ENTITY (which NF provides / is responsible for a service or operation)
      ≥2 term → FG   (NF↔NF interaction: steps ∪ service-operation bridge)
      1 term  → A1
      0 term  → M / P / C

  WHY_COND  (condition / reason / when / requirement)
      1 term  → A1   (requirement chunk_type ranked up inside A1's CASE ladder)
      ≥2 term → A2
      0 term  → B / C

Only WH-types that can carry a message/parameter/interface slot (WHAT_DEF with 0
terms, HOW_PROC, WHERE_LOC, WHO_ENTITY) invoke the LLM slot extractor — the
common term-anchored cases (FUNC/A1/A2/F/G) need no LLM at all.
"""
from __future__ import annotations

from typing import Optional

# intent (lowercased) → WH-type. Anything unmapped falls to WHAT_DEF (the most
# forgiving tree: it degrades to A1/A2/C which the vector branch backstops).
_INTENT_TO_WH = {
    # WHAT — definition / functionality / capability
    "definition": "WHAT_DEF",
    "what_is": "WHAT_DEF",
    "abbreviation": "WHAT_DEF",
    "network_function": "WHAT_DEF",
    "capability": "WHAT_DEF",
    "list": "WHAT_DEF",
    "general": "WHAT_DEF",
    "overview": "WHAT_DEF",
    # HOW — procedure / mechanism / call-flow
    "procedure": "HOW_PROC",
    "how_does": "HOW_PROC",
    "how_to": "HOW_PROC",
    "mechanism": "HOW_PROC",
    "call_flow": "HOW_PROC",
    # WHERE — location / spec / interface
    "location": "WHERE_LOC",
    "reference": "WHERE_LOC",
    "which_spec": "WHERE_LOC",
    "interface": "WHERE_LOC",
    "cross_spec": "WHERE_LOC",
    # WHO — provider / entity
    "provider": "WHO_ENTITY",
    "which_nf": "WHO_ENTITY",
    "entity": "WHO_ENTITY",
    "relationship": "WHO_ENTITY",
    # WHY — condition / reason
    "reason": "WHY_COND",
    "condition": "WHY_COND",
    "when": "WHY_COND",
    "requirement": "WHY_COND",
}

WH_TYPES = ("WHAT_DEF", "HOW_PROC", "WHERE_LOC", "WHO_ENTITY", "WHY_COND")

# WH-types whose tree may need an LLM-extracted slot (message/parameter/interface)
# when no term resolved. The term-anchored trees never call the LLM. WHY_COND is
# included so its 0-term interface branch (B) can actually fire.
_WH_NEEDS_SLOT = {"WHAT_DEF", "HOW_PROC", "WHERE_LOC", "WHO_ENTITY", "WHY_COND"}


def classify_wh(intent: Optional[str]) -> str:
    """Tier 1: map intent → WH-type. Deterministic, no LLM."""
    return _INTENT_TO_WH.get((intent or "").strip().lower(), "WHAT_DEF")


def needs_slot_extraction(wh_type: str, n_terms: int) -> bool:
    """True when Tier 2 could resolve to a slot-bearing pattern (M/P/B/E).
    ALWAYS extract now (2026-07-12 regression fix): a named message/parameter can
    appear in a question that ALSO carries terms — "how does LOCATION REPORT differ
    between NGAP, S1AP, RANAP" names a message AND three protocol terms. The old
    `n_terms == 0` gate skipped extraction for exactly those M/P questions, so they
    fell through to A2 and lost all Pattern M/P recall. The slot extractor returns
    all-null for a pure NF↔NF question, so extracting unconditionally is safe — it
    just costs one small LLM call, and per goal.md latency is not the constraint."""
    return True


def route(
    wh_type: str,
    n_terms: int,
    has_spec_ref: bool,
    slots: Optional[dict] = None,
) -> str:
    """Tier 2: pick the concrete Cypher pattern id for a WH-type. `slots` carries
    the optionally-LLM-extracted message_name / parameter_name / interface_id so
    the tree can prefer a named-entity pattern over the sentinel. Returns a
    pattern id in cypher_templates.VALID_PATTERNS."""
    slots = slots or {}
    msg = slots.get("message_name")
    param = slots.get("parameter_name")
    iface = slots.get("interface_id")

    # Named-entity slots take PRECEDENCE over term count (2026-07-12 regression
    # fix). A question naming a specific ALL-CAPS message ("LOCATION REPORT") or a
    # named IE/parameter ("Transport Layer Address") is an M/P question even when
    # it also mentions protocol terms (NGAP/S1AP/RANAP, which resolve to Terms).
    # The old order ("n_terms>=2 → A2/FG" first) routed every M/P question that
    # carried protocol terms to A2, wiping out Pattern M/P recall on wh_kg_v3_100
    # (M 0.61→0.08, P 0.50→0.07). The message/parameter name is a far stronger
    # signal of intent than how many terms happen to be in the sentence.
    if msg:
        return "M"
    if param:
        return "P"

    if wh_type == "WHAT_DEF":
        if n_terms >= 2:
            return "A2"
        if n_terms == 1:
            return "FUNC"
        if iface:
            return "B"
        return "C"

    if wh_type == "HOW_PROC":
        if n_terms >= 2:
            return "FG"
        if n_terms == 1:
            return "A1"
        if iface:
            return "B"
        return "C"

    if wh_type == "WHERE_LOC":
        if has_spec_ref:
            return "E"
        if iface:
            return "B"
        if n_terms >= 2:
            return "A2"
        if n_terms == 1:
            return "A1"
        return "C"

    if wh_type == "WHO_ENTITY":
        if n_terms >= 2:
            # ≥2 NFs with a 'relationship'/provider intent is still an NF↔NF
            # interaction — take the FG union (steps ∪ service-operation) rather
            # than betting on G alone (G-005: some gold is step-side, not
            # operation-side; the internal rerank narrows it either way).
            return "FG"
        if n_terms == 1:
            return "A1"
        if msg:
            return "M"
        if param:
            return "P"
        return "C"

    if wh_type == "WHY_COND":
        if n_terms >= 2:
            return "A2"
        if n_terms == 1:
            return "A1"
        if iface:
            return "B"
        return "C"

    # Unknown WH-type — safest fallback.
    return "C"
