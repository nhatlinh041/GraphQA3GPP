"""
Term-First strategy — assemble the structured term dict that downstream
retrieval components consume. There are two paths:

1. extract_with_llm_terms() — preferred. Caller already has LLM-extracted
   abbreviations / full_names / spec_refs (from IntentClassifier.classify_with_terms).
   We hard-validate each candidate against the live TermIndex and drop
   anything not in the KG (prevents Cypher hallucinations like
   `MATCH (t:Term {abbreviation: 'TELL'})`).

2. extract_fallback() — used when the LLM call failed (timeout, parse error).
   Pure deterministic path: full_name regex match (longest-first) + simple
   token-based abbreviation lookup against the same TermIndex. No hard-coded
   lists — KG is the source of truth.

Output shape (stable contract for downstream):
    {
        'network_functions': list[str],   # uppercase abbreviations validated against KG
        'spec_refs':         list[str],   # ['TS 23.501', ...]
        'abbreviations':     list[str],   # extra abbrevs not already in network_functions
        'primary_term':      str,         # best single anchor for graph lookup
        'resolved':          dict,        # {abbrev: {full_name, specs, matched_property}}
    }
"""
import re

from .term_index import TermIndex


# Pattern for spec references like "TS 23.501" / "TR 38.300" — applied to
# both LLM-emitted spec_refs and the fallback regex path. Case-insensitive.
SPEC_PATTERN = re.compile(r"\b(TS|TR)\s*(\d{2}\.\d{3})\b", re.IGNORECASE)

# Tokeniser for the fallback path: pulls candidate identifiers from a query.
# Lowercase tokens accepted — TermIndex normalises case on lookup. The 1-15
# char range covers everything from 2-letter ("AF", "AI") to long compounds
# in the KG without slipping in stop words too aggressively.
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9-]{1,14}")


class TermFirstStrategy:
    def __init__(self, index: TermIndex):
        self._index = index

    # Path 1: LLM extracted candidates → validate against KG → assemble.
    def extract_with_llm_terms(
        self,
        query: str,
        llm_abbreviations: list[str],
        llm_full_names: list[str],
        llm_spec_refs: list[str],
    ) -> dict:
        nfs: list[str] = []
        resolved: dict[str, dict] = {}

        # Track full-name matches separately so primary_term can prefer them
        # (a verbatim full-name mention in the query is the strongest signal).
        full_name_hits: list[str] = []  # canonical abbreviations from full_name path

        # Validate full_names first — they are the strongest signal and we
        # want their abbreviation to win the primary_term tiebreaker.
        for name in llm_full_names:
            rec = self._index.lookup_full_name(name)
            if rec is None:
                continue
            abbrev = rec["abbreviation"]
            if abbrev not in resolved:
                resolved[abbrev] = _resolved_entry(rec)
                nfs.append(abbrev)
            full_name_hits.append(abbrev)

        # Validate abbreviations — drop hallucinations (anything not in KG).
        for abbrev in llm_abbreviations:
            rec = self._index.lookup_abbrev(abbrev)
            if rec is None:
                continue
            canon = rec["abbreviation"]
            if canon not in resolved:
                resolved[canon] = _resolved_entry(rec)
                nfs.append(canon)

        # spec_refs: re-validate format (LLM may emit "Ts23.501" or
        # "TS-23.501" inconsistently). Canonicalise to "TS 23.501" / "TR 38.300".
        spec_refs = _normalise_spec_refs(llm_spec_refs)

        primary = _pick_primary(
            full_name_hits=full_name_hits,
            abbreviations=nfs,
            spec_refs=spec_refs,
        )

        return _assemble(
            network_functions=nfs,
            spec_refs=spec_refs,
            abbreviations=[],  # no "extra" bucket needed when LLM is the source
            primary=primary,
            resolved=resolved,
        )

    # Path 2: deterministic fallback when the LLM is unavailable. Uses the
    # same TermIndex so behaviour is consistent — just no semantic understanding.
    def extract_fallback(self, query: str) -> dict:
        nfs: list[str] = []
        resolved: dict[str, dict] = {}
        full_name_hits: list[str] = []

        # Strongest signal first: full_name regex match (longest-first).
        for abbrev, _ in self._index.find_full_name_matches(query):
            rec = self._index.lookup_abbrev(abbrev)
            if rec is None:
                continue
            if abbrev not in resolved:
                resolved[abbrev] = _resolved_entry(rec)
                nfs.append(abbrev)
            full_name_hits.append(abbrev)

        # Then per-token abbreviation lookup. TermIndex.lookup_abbrev uppercases
        # internally — accept tokens in any case. KG-not-found tokens (common
        # English words, "tell", "what", ...) silently get None.
        for tok in TOKEN_PATTERN.findall(query):
            rec = self._index.lookup_abbrev(tok)
            if rec is None:
                continue
            canon = rec["abbreviation"]
            if canon not in resolved:
                resolved[canon] = _resolved_entry(rec)
                nfs.append(canon)

        spec_refs = _scan_spec_refs(query)

        primary = _pick_primary(
            full_name_hits=full_name_hits,
            abbreviations=nfs,
            spec_refs=spec_refs,
        )

        return _assemble(
            network_functions=nfs,
            spec_refs=spec_refs,
            abbreviations=[],
            primary=primary,
            resolved=resolved,
        )


# ---- helpers --------------------------------------------------------------


# Build the {full_name, specs, matched_property} record consumed by
# cypher_generator + adaptive_hop. Mirrors the legacy TermResolver shape.
def _resolved_entry(rec: dict) -> dict:
    out = {
        "full_name": rec.get("full_name"),
        "specs": list(rec.get("source_specs") or []),
        "matched_property": "abbreviation",
    }
    # Entity outside :Term (Message/ServiceOperation/Concept/Parameter) — cypher_generator
    # needs it to MATCH the right label, otherwise it emits (t:Term {abbreviation:'<message name>'})
    # and always gets 0 rows.
    if rec.get("node_label"):
        out["node_label"] = rec["node_label"]
    return out


# Pick a single anchor for graph lookup. Order:
# 1. First full_name hit (strongest — verbatim mention).
# 2. First abbreviation hit (validated against KG).
# 3. First spec ref.
# Returns "" when none resolved — do NOT fall back to a raw query token here.
# An unvalidated word (e.g. the query's first word, "For") is indistinguishable
# from a real KG-validated term to every downstream consumer (cypher_generator's
# "Primary term hint: X" prompt block, orchestrator's react_agent seed list), so
# it was silently anchoring Cypher Pattern A1 to nonexistent Term nodes like
# `MATCH (t:Term {abbreviation: 'For'})` — always 0 rows. An empty primary_term
# correctly falls through to Pattern B/D/C instead (see cypher_generator._build_prompt).
def _pick_primary(
    full_name_hits: list[str],
    abbreviations: list[str],
    spec_refs: list[str],
) -> str:
    if full_name_hits:
        return full_name_hits[0]
    if abbreviations:
        return abbreviations[0]
    if spec_refs:
        return spec_refs[0]
    return ""


# Normalise LLM-emitted spec refs to canonical "TS XX.XXX" / "TR XX.XXX".
def _normalise_spec_refs(refs: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        m = SPEC_PATTERN.search(ref or "")
        if not m:
            continue
        canon = f"{m.group(1).upper()} {m.group(2)}"
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


# Scan an arbitrary string for "TS 23.501" / "TR 38.300" references.
def _scan_spec_refs(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in SPEC_PATTERN.finditer(text):
        canon = f"{m.group(1).upper()} {m.group(2)}"
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


# Final dict assembly — keeps the contract stable so orchestrator/cypher_generator
# don't need to know which extraction path produced the result.
def _assemble(
    network_functions: list[str],
    spec_refs: list[str],
    abbreviations: list[str],
    primary: str,
    resolved: dict,
) -> dict:
    return {
        "network_functions": network_functions,
        "spec_refs": spec_refs,
        "abbreviations": abbreviations,
        "primary_term": primary,
        "resolved": resolved,
    }
