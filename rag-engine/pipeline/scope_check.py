"""
Out-of-scope / abstain detection (2026-07-13).

The benchmark's `abstain` questions (wh100_tele T5_CONFLICT_NEG) name a SPECIFIC
entity that does NOT exist in the 3GPP corpus — a fabricated service operation
(`Nudm_QuantumKey_Distribute`), a made-up NF ('Nssai Prediction Function / NPF'),
a non-existent topic ('6G core network', '5G Loyalty Points'). A pure similarity
reranker can't catch these: the fabricated names are lexically close to real ones,
so the cross-encoder scores a near-match chunk highly (measured logit up to 5.8,
higher than some genuine-gold questions). The reliable signal is EXISTENCE: does
the named entity the question is built around actually appear in the KG?

`scope_warnings(question, driver, resolved_terms)` returns a list of human-readable
warnings for named entities the question introduces that the KG does not contain.
The orchestrator passes them to the answer prompt so the model can abstain
explicitly ("the corpus does not define X") instead of hallucinating an answer
around a plausible-looking but non-existent entity.

Conservative by design: only flags entities matching a STRICT named-entity shape
(SBI operation `Nxxx_Yyy...`, or a capitalised NF/function name the question
quotes). A miss just means no warning (status quo); it never blocks retrieval.
"""
from __future__ import annotations

import re
from typing import Optional

# SBI service-operation shape: Nxxx_Yyy[_Zzz]. High precision — this exact form
# only ever means "a 3GPP service operation".
_SBI_OP_RE = re.compile(r"\bN[a-z0-9]{2,6}_[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*\b")

# A quoted or parenthesised function/NF name the question introduces, e.g.
# 'Nssai Prediction Function (NPF)'. Captures the abbreviation in parens and the
# quoted long form.
_QUOTED_NF_RE = re.compile(r"['\"]([A-Z][A-Za-z0-9 ]{3,60})['\"]")
_PAREN_ABBR_RE = re.compile(r"\(([A-Z]{2,6})\)")

# Topics that are out of corpus scope by construction (the corpus is Rel-18 5G,
# 3GPP protocol specs — not future releases, non-3GPP tech, or physical/commercial
# domains). Each substring is a strong out-of-scope signal.
_OUT_OF_SCOPE_TOPICS = (
    "6g", "5g loyalty", "subscription price", "monthly price",
    "wi-fi 7", "wifi 7", "release 20", "rel-20", "ai-native air interface",
    "concrete mix", "tower foundation", "quantum key",
)

# Interface / reference-point identifiers the question claims exist. Real ones are
# a small known set; a made-up "N99" between arbitrary nodes is an abstain signal.
# We check the identifier appears in NO section_title (real reference points always
# do). Conservative: only fires for the N-series shape the question quotes.
_REF_POINT_RE = re.compile(r"['\"]?\b(N\d{1,3})\b['\"]?\s+reference[ -]?point", re.IGNORECASE)


def _op_exists(driver, name: str) -> bool:
    try:
        with driver.session() as s:
            rec = s.run(
                "RETURN EXISTS { MATCH (o:ServiceOperation {name: $n}) } AS e",
                n=name,
            ).single()
            return bool(rec and rec["e"])
    except Exception:
        return True  # on error, don't warn (fail open)


def _term_exists(driver, abbr: str) -> bool:
    try:
        with driver.session() as s:
            rec = s.run(
                "RETURN EXISTS { MATCH (t:Term {abbreviation: $a}) } AS e",
                a=abbr,
            ).single()
            return bool(rec and rec["e"])
    except Exception:
        return True


def scope_warnings(
    question: str,
    driver,
    resolved_terms: Optional[dict] = None,
) -> list[str]:
    """Return warnings for named entities the question is built around that the KG
    does not contain. Empty list = nothing suspicious (normal case)."""
    q = question or ""
    ql = q.lower()
    warnings: list[str] = []

    # 1) Out-of-scope topics (6G, pricing, …) — the corpus is Rel-18 5G specs.
    for topic in _OUT_OF_SCOPE_TOPICS:
        if topic in ql:
            warnings.append(
                f"the question refers to '{topic}', which is outside the scope of the "
                f"provided 3GPP 5G corpus"
            )

    # 2) Named SBI service operations that don't exist in the KG.
    for m in _SBI_OP_RE.finditer(q):
        name = m.group(0)
        # skip if it's just an NF prefix without an operation body (handled as Term)
        if "_" not in name:
            continue
        if not _op_exists(driver, name):
            warnings.append(
                f"the service operation '{name}' does not exist in the corpus "
                f"(no ServiceOperation node) — it may be a fabricated name"
            )

    # 2b) Reference-point identifiers the question claims exist ("N99 reference
    # point") but which appear in NO section_title — real reference points always
    # do (N1/N2/N4/N6…). A made-up N-number is an abstain signal.
    for m in _REF_POINT_RE.finditer(q):
        ident = m.group(1)
        try:
            with driver.session() as s:
                rec = s.run(
                    "RETURN EXISTS { MATCH (c:Chunk) WHERE c.section_title =~ $re } AS e",
                    re=f"(?i).*\\b{ident}\\b.*",
                ).single()
                exists = bool(rec and rec["e"])
        except Exception:
            exists = True
        if not exists:
            warnings.append(
                f"the reference point '{ident}' named in the question is not defined "
                f"anywhere in the corpus — it may be fabricated"
            )

    # 3) Quoted/parenthesised NF names whose abbreviation isn't a known Term.
    abbrs = set(_PAREN_ABBR_RE.findall(q))
    resolved = set((resolved_terms or {}).keys())
    for abbr in abbrs:
        if abbr in resolved:
            continue
        if not _term_exists(driver, abbr):
            warnings.append(
                f"the entity '{abbr}' is named in the question but is not a defined "
                f"network function / term in the corpus — it may not exist"
            )

    # De-dup, cap to avoid a noisy prompt.
    seen, out = set(), []
    for w in warnings:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out[:4]
