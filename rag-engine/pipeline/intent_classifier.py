"""
Intent classifier — calls the LLM (same model the user chose) to map a question
into one of the intents + return JSON. Has a regex fallback so the pipeline keeps
working when the LLM fails (timeout, parse error, model returns an out-of-enum intent).
"""
import json
import re
from typing import Optional

from llm import OllamaClient


# Intent enum — must match what cypher_generator + orchestrator understand
ALLOWED_INTENTS = (
    "definition",
    "procedure",
    "comparison",
    "network_function",
    "reference",
    "relationship",
    "general",
)


# Regex pattern fallback — used when the LLM is unavailable or returns invalid output.
# Priority order kept unchanged so behavior stays predictable in offline tests.
_REGEX_PATTERNS: dict[str, list[str]] = {
    "definition": [
        r"\bwhat is\b", r"\bwhat are\b", r"\bdefine\b", r"\bmeaning of\b",
        r"\bstand for\b", r"\bacronym\b",
    ],
    "procedure": [
        r"\bhow does\b", r"\bhow to\b", r"\bsteps\b", r"\bprocess\b",
        r"\bprocedure\b", r"\bflow\b", r"\bsequence\b",
    ],
    "comparison": [
        r"\bdifference\b", r"\bcompare\b", r"\bvs\b", r"\bversus\b",
        r"\bsimilar\b", r"\bdistinguish\b",
    ],
    "network_function": [
        r"\bamf\b", r"\bsmf\b", r"\bupf\b", r"\bnrf\b", r"\bausf\b",
        r"\budm\b", r"\bpcf\b", r"\bnssf\b", r"\baf\b", r"\bnef\b",
        r"\bsepp\b", r"\bscp\b",
    ],
    "reference": [
        r"\bts\s+\d{2}\.\d{3}\b", r"\bspec\b", r"\bspecification\b",
        r"\bdocument\b", r"\bclause\b", r"\bsection\b",
    ],
    "relationship": [
        r"\brelationship\b", r"\binterface\b", r"\binteract\b",
        r"\bconnect\b", r"\blink\b", r"\bbetween\b.*\band\b",
    ],
}


# LLM prompt — short, with disambiguation rules for ambiguous cases (e.g. "What is
# AMF?" matches both definition and network_function); Ollama format=json ensures parse OK.
def _build_prompt(query: str) -> str:
    return f"""Classify this 3GPP technical question into exactly ONE intent.

Intents:
- definition: "What is X?", "Define X", "Meaning of X" — asking for a concept's meaning.
- procedure: "How does X work?", "Steps to do X" — asking for a process or sequence.
- comparison: "Difference between X and Y", "Compare X vs Y" — asking how things differ.
- network_function: questions whose PRIMARY focus is listing/attributes of a 5G NF (AMF/SMF/UPF/NRF/...) where the question is NOT a what/how/compare/relationship form.
- reference: questions about which spec/clause/document defines something.
- relationship: "How is X related to Y?", "What NFs interact with X?", "Interface between X and Y".
- general: anything else.

Disambiguation rules (apply in order):
1. If the question matches "What is X?" / "Define X" — choose `definition`, even if X is a network function.
2. If the question is "List/which NFs interact with X" or "How does X talk to Y" — choose `relationship`.
3. If the question is purely about a NF without what/how/compare/relationship verbs — choose `network_function`.

Question: {query}

Respond with STRICT JSON, no prose, no markdown fences:
{{"intent": "<one of: definition|procedure|comparison|network_function|reference|relationship|general>"}}
"""


# Combined prompt: same intent classification + entity extraction in one LLM call.
# Returned terms are validated downstream against the live KG TermIndex,
# so hallucinated abbreviations / full_names are filtered out before reaching
# the Cypher generator. Phrasing emphasises "extract what user wrote" to keep
# the LLM honest.
def _build_prompt_with_terms(query: str) -> str:
    return f"""Classify this 3GPP technical question AND extract any 3GPP entities mentioned.

Intents:
- definition: "What is X?", "Define X", "Meaning of X" — asking for a concept's meaning.
- procedure: "How does X work?", "Steps to do X" — asking for a process or sequence.
- comparison: "Difference between X and Y", "Compare X vs Y" — asking how things differ.
- network_function: questions whose PRIMARY focus is listing/attributes of a 5G NF (AMF/SMF/UPF/NRF/...) where the question is NOT a what/how/compare/relationship form.
- reference: questions about which spec/clause/document defines something.
- relationship: "How is X related to Y?", "What NFs interact with X?", "Interface between X and Y".
- general: anything else.

Disambiguation rules (apply in order):
1. If the question matches "What is X?" / "Define X" — choose `definition`, even if X is a network function.
2. If the question is "List/which NFs interact with X" or "How does X talk to Y" — choose `relationship`.
3. If the question is purely about a NF without what/how/compare/relationship verbs — choose `network_function`.

Entity extraction rules:
- Extract abbreviations (e.g. "SCP", "AMF", "UPF") even if user wrote them lowercase. Always emit them UPPERCASE.
- Extract full names of network functions / services / interfaces / proxies (e.g. "Service Communication Proxy", "Session Management Function", "Access and Mobility Management Function").
- Extract ALL-CAPS protocol message names VERBATIM, keeping every word (e.g. "DEDICATED MEASUREMENT FAILURE INDICATION", "PATH SWITCH REQUEST", "RAN CONFIGURATION UPDATE FAILURE"). Put them in full_names, not abbreviations.
- Extract SBI service operation names of the form Nxxx_Yyy_Zzz verbatim (e.g. "Nnef_ECSAddress_Create", "Nudm_UECM_Get"). Put them in full_names.
- Extract information element / attribute / parameter names as written (e.g. "DAPS Indicator", "Traffic Profile", "ueIpv4Addr", "sNssaiList"). Put them in full_names.
- Extract standardized value or enumeration names as written (e.g. "5QI", "PQI", "AccessType", "DataSetId").
- Extract spec references in the form "TS XX.XXX" or "TR XX.XXX" (e.g. "TS 23.501").
- DO NOT invent. Extract ONLY what the user wrote (lowercased forms accepted, but emit abbreviations uppercased).
- DO NOT extract generic words like "5G", "4G", "LTE", "NR", "network", "system", "core", "function".
- If the user wrote a full name, also include its abbreviation if you are SURE it is the standard 3GPP abbreviation (e.g. "Service Communication Proxy" → "SCP"). When unsure, leave it out — it will still be matched downstream.

Question: {query}

Respond with STRICT JSON, no prose, no markdown fences:
{{
  "intent": "<one of: definition|procedure|comparison|network_function|reference|relationship|general>",
  "abbreviations": ["<UPPERCASE abbrev>", ...],
  "full_names": ["<full name as written>", ...],
  "spec_refs": ["TS XX.XXX", ...]
}}
"""


# Defensive parser: LLM may return None / a non-list / nested junk. Always
# return list[str] with empty strings filtered, no exceptions.
def _coerce_str_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
    return out


class IntentClassifier:
    """LLM-first classifier with regex fallback. Constructor takes an OllamaClient
    so classification uses the orchestrator's same LLM instance (keeps Ollama warm)."""

    def __init__(self, llm: Optional[OllamaClient] = None):
        self._llm = llm

    def classify(
        self, query: str, model: Optional[str] = None, num_ctx: Optional[int] = None
    ) -> str:
        """LLM path if llm + model are present; fall back to regex on any failure.
        Returns an intent string (always within ALLOWED_INTENTS).
        `num_ctx` MUST match the value the rest of the request uses — see
        `classify_with_terms`."""
        if self._llm is not None and model:
            llm_intent = self._classify_llm(query, model, num_ctx=num_ctx)
            if llm_intent is not None:
                return llm_intent
        return self._classify_regex(query)

    # Combined intent + term extraction in one LLM call. Returns None on any
    # failure (timeout, parse error, invalid intent) so the caller can fall
    # back to (regex intent + TermIndex regex extraction).
    # Returned shape:
    #   {"intent": str, "abbreviations": list[str], "full_names": list[str],
    #    "spec_refs": list[str]}
    # Terms are NOT validated against the KG here — that is the caller's job
    # via TermIndex (hard-validate). This keeps the classifier purely
    # presentation/parsing layer.
    # `num_ctx` must be the SAME value every other Ollama call in the request uses
    # (cypher-gen, answer stream). Ollama keys a loaded runner by (model, options):
    # a different num_ctx means a different KV-cache size, so it kills the running
    # llama-server and reloads the full ~9GB of weights. Omitting it here made
    # Ollama fall back to its 4096 default while the rest of the pipeline asked for
    # 8192 → two full model reloads per question.
    def classify_with_terms(
        self, query: str, model: Optional[str] = None, num_ctx: Optional[int] = None
    ) -> Optional[dict]:
        if self._llm is None or not model:
            return None
        try:
            raw = self._llm.generate(
                _build_prompt_with_terms(query),
                model=model,
                format="json",
                think=False,
                timeout=90,
                num_ctx=num_ctx,
            )
        except Exception:
            return None
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(obj, dict):
            return None

        intent = str(obj.get("intent", "")).strip().lower()
        if intent not in ALLOWED_INTENTS:
            return None

        # Defensive parsing — accept missing keys / wrong types as empty.
        abbreviations = _coerce_str_list(obj.get("abbreviations"))
        full_names = _coerce_str_list(obj.get("full_names"))
        spec_refs = _coerce_str_list(obj.get("spec_refs"))

        # Normalize abbreviations to UPPER (KG canonical) — LLM might still
        # emit them lowercase despite the prompt instruction.
        abbreviations = [a.upper() for a in abbreviations if a]

        return {
            "intent": intent,
            "abbreviations": abbreviations,
            "full_names": full_names,
            "spec_refs": spec_refs,
        }

    # Call LLM with format=json + think=False (intent is a small task, no CoT needed).
    # Returns an intent string if valid, None on failure (so caller falls back to regex).
    def _classify_llm(
        self, query: str, model: str, num_ctx: Optional[int] = None
    ) -> Optional[str]:
        try:
            raw = self._llm.generate(
                _build_prompt(query),
                model=model,
                format="json",
                think=False,
                # 90s timeout covers cold-start qwen3:14b (~45s); subsequent
                # calls with the model warm take only 1-3s.
                timeout=90,
                num_ctx=num_ctx,
            )
        except Exception:
            return None
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(obj, dict):
            return None
        intent = str(obj.get("intent", "")).strip().lower()
        return intent if intent in ALLOWED_INTENTS else None

    # Original regex classifier — kept as a safety net.
    def _classify_regex(self, query: str) -> str:
        q = query.lower()

        # Network function check first (high precision) — preserves old matching.
        for pattern in _REGEX_PATTERNS["network_function"]:
            if re.search(pattern, q):
                return "network_function"

        # Other intents by match count
        scores: dict[str, int] = {}
        for intent, patterns in _REGEX_PATTERNS.items():
            if intent == "network_function":
                continue
            scores[intent] = sum(1 for p in patterns if re.search(p, q))

        if not scores or max(scores.values()) == 0:
            return "general"

        return max(scores, key=lambda i: scores[i])
