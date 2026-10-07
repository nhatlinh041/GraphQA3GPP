# -*- coding: utf-8 -*-
"""Per-request ablation switches.

Three retrieval switches used to be module-level constants read once from the
environment at import time, which meant changing one required a whole new server
process. run_bench worked around that by starting a private uvicorn per config —
and that workaround is what produced the port-squatting bug recorded in CLAUDE.md,
where a leftover instance answered the health probe and eight configs silently ran
with the wrong environment.

They are ContextVars rather than plain parameters because the flags are consumed
four call levels below the entry point (fusion._rerank_with_dedup has four call
sites of its own) and threading a keyword through all of them is where a default
gets dropped on one path and nobody notices. asyncio.to_thread copies the current
context, so the worker threads this pipeline uses see the same values.

The environment variables still work and still mean the same thing: an unset
override falls back to them, so existing run_bench configs and any deployment that
sets them behave exactly as before.
"""
from __future__ import annotations

import os
from contextvars import ContextVar

_TRUE = ("1", "true", "True")


def _env(name: str) -> bool:
    return os.getenv(name, "0") in _TRUE


# Process-wide defaults, read once — the previous behaviour.
ENV_TERM_EXPANSION_OFF = _env("TERM_EXPANSION_OFF")
ENV_RERANK_OFF = _env("RERANK_OFF")
ENV_TITLE_SEARCH_OFF = _env("TITLE_SEARCH_OFF")
ENV_KG_ABLATION = os.getenv("KG_ABLATION", "off").lower()
ENV_KG_EVIDENCE = os.getenv("KG_EVIDENCE", "text").lower()
ENV_PROVENANCE_ON = os.getenv("PROVENANCE_ON", "1") not in ("0", "false", "")

# Layer order for the KG ablation ladder: each level adds the edge family named.
KG_ABL_ORDER = ["entity", "relation", "hierarchy", "xref", "provenance"]

# None means "not overridden for this request" — NOT False. The distinction
# matters: a request that omits the field must inherit the environment default,
# not silently turn the ablation off for a server started with it on.
_term: ContextVar[bool | None] = ContextVar("term_expansion_off", default=None)
_rerank: ContextVar[bool | None] = ContextVar("rerank_off", default=None)
_title: ContextVar[bool | None] = ContextVar("title_search_off", default=None)
_kgabl: ContextVar[str | None] = ContextVar("kg_ablation", default=None)
_kgev: ContextVar[str | None] = ContextVar("kg_evidence", default=None)
_prov: ContextVar[bool | None] = ContextVar("provenance_on", default=None)


def set_overrides(*, term_expansion_off: bool | None = None,
                  rerank_off: bool | None = None,
                  title_search_off: bool | None = None,
                  kg_ablation: str | None = None,
                  kg_evidence: str | None = None,
                  provenance_on: bool | None = None) -> None:
    """Called once per request at the orchestrator entry point."""
    _term.set(term_expansion_off)
    _rerank.set(rerank_off)
    _title.set(title_search_off)
    _kgabl.set(kg_ablation)
    _kgev.set(kg_evidence)
    _prov.set(provenance_on)


def term_expansion_off() -> bool:
    v = _term.get()
    return ENV_TERM_EXPANSION_OFF if v is None else v


def rerank_off() -> bool:
    v = _rerank.get()
    return ENV_RERANK_OFF if v is None else v


def title_search_off() -> bool:
    v = _title.get()
    return ENV_TITLE_SEARCH_OFF if v is None else v


def kg_ablation() -> str:
    v = _kgabl.get()
    return ENV_KG_ABLATION if v is None else v


def kg_evidence() -> str:
    v = _kgev.get()
    return ENV_KG_EVIDENCE if v is None else v


def provenance_on() -> bool:
    v = _prov.get()
    return ENV_PROVENANCE_ON if v is None else v


def kg_layer_on(feature: str) -> bool:
    """True if `feature`'s edge family is active at the current ablation level.
    An unset or unrecognised level disables the gate entirely, so the live system
    behaves as if no ablation were configured."""
    lvl = kg_ablation()
    if lvl not in KG_ABL_ORDER:
        return True
    return KG_ABL_ORDER.index(lvl) >= KG_ABL_ORDER.index(feature)


def active() -> dict:
    """What is in force for this request — emitted into the trace so a run can be
    audited after the fact instead of trusting the config table."""
    return {"term_expansion_off": term_expansion_off(),
            "rerank_off": rerank_off(),
            "title_search_off": title_search_off(),
            "kg_ablation": kg_ablation(),
            "kg_evidence": kg_evidence(),
            "provenance_on": provenance_on()}
