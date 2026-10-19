"""
Deterministic Cypher template builder (2026-07-12).

Splits Cypher generation into TWO steps to remove the non-determinism of the
single "pick a pattern AND write the query" LLM call (observed: Pattern G chosen
correctly but the emitted Cypher varied run-to-run — sometimes returning the gold
chunk, sometimes not, sometimes empty):

  Step 1 (LLM, tiny): choose ONE pattern id (A1/A2/B/D/E/F/G/M/P/C) — see
          cypher_generator's pattern-selection prompt.
  Step 2 (THIS module, deterministic): fill the chosen pattern's fixed template
          from already-resolved slots (term abbreviations from intent extraction,
          message/parameter names matched against the KG, spec_ids, etc.).

Every template here is the SAME query the old prompt asked the LLM to write, but
now emitted byte-for-byte identically, so a given (pattern, slots) always yields
the same result set. `build_cypher` returns None when the chosen pattern's
required slots are missing — the caller then falls back to the LLM path.
"""
from __future__ import annotations

import re
from typing import Optional

# Intent → chunk_type list (mirrors the mapping documented in the prompt).
_INTENT_CHUNK_TYPES = {
    "definition": ["definition", "abbreviation"],
    "what_is": ["definition", "abbreviation"],
    "abbreviation": ["definition", "abbreviation"],
    "network_function": ["definition", "abbreviation"],
    "procedure": ["procedure", "requirement"],
    "how_does": ["procedure", "requirement"],
    "general": ["definition", "general"],
    "overview": ["definition", "general"],
}
_DEFAULT_CHUNK_TYPES = ["definition", "general"]


def _chunk_type_list(intent: Optional[str]) -> str:
    types = _INTENT_CHUNK_TYPES.get((intent or "").lower(), _DEFAULT_CHUNK_TYPES)
    return "[" + ", ".join(f"'{t}'" for t in types) + "]"


def _abbr_list(abbrevs: list[str]) -> str:
    # Abbreviations are KG-canonical UPPER already; quote for the IN list.
    return "[" + ", ".join(f"'{a}'" for a in abbrevs) + "]"


def _esc(s: str) -> str:
    """Escape a single-quoted Cypher string literal (backslash + apostrophe)."""
    return s.replace("\\", "\\\\").replace("'", "\\'")


# ── Templates ────────────────────────────────────────────────────────────────
# Each returns a complete single-statement Cypher string, or None if a required
# slot is absent. Column contract: chunk_id, content, spec_id, section, score.

def _tpl_a1(abbrevs: list[str], intent: Optional[str], **_) -> Optional[str]:
    if len(abbrevs) < 1:
        return None
    a = _esc(abbrevs[0])
    return f"""MATCH (t:Term {{abbreviation: '{a}'}})
WITH t, t.full_name AS full_name LIMIT 1
MATCH (c:Chunk)
WHERE (c.spec_id IN t.source_specs AND c.chunk_type IN {_chunk_type_list(intent)})
   OR (full_name IS NOT NULL AND c.section_title CONTAINS full_name)
OPTIONAL MATCH (c)-[m:MENTIONS]->(t)
WITH c, full_name, m,
  CASE
    WHEN full_name IS NOT NULL AND c.section_title CONTAINS full_name THEN 1.0
    WHEN m IS NOT NULL                 THEN 0.92
    WHEN c.chunk_type = 'definition'   THEN 0.85
    WHEN c.chunk_type = 'abbreviation' THEN 0.8
    WHEN c.chunk_type = 'procedure'    THEN 0.8
    WHEN c.chunk_type = 'requirement'  THEN 0.75
    ELSE 0.6
  END AS score
RETURN c.chunk_id AS chunk_id, c.content AS content, c.spec_id AS spec_id,
       c.section_title AS section, score
ORDER BY score DESC LIMIT $top_k"""


def _tpl_func(abbrevs: list[str], **_) -> Optional[str]:
    """FUNC — "what are the functionalities of <term>". Goes term → the spec that
    DEFINES it (source_specs) → the chunk whose SECTION is about the term itself
    (section_title contains the full_name or equals the abbreviation) and is a
    functional chunk_type, NOT Scope/References/Abbreviations. This is the
    structured answer to the "SCP functionalities" case: instead of pulling every
    chunk of every source_spec (Pattern A1) and hoping rerank sorts it out, we
    only surface chunks that are ACTUALLY about the term. Score tiers:
      1.0  section_title == abbreviation, or contains full_name  (the term's OWN section)
      0.85 functional chunk_type (definition/general/requirement/overview) that MENTIONS it
    Requires exactly the term slot; caller uses it for single-term definition/
    capability questions."""
    if len(abbrevs) < 1:
        return None
    a = _esc(abbrevs[0])
    return f"""MATCH (t:Term {{abbreviation: '{a}'}})
WITH t, t.full_name AS full_name, t.abbreviation AS abbr LIMIT 1
MATCH (c:Chunk)-[:MENTIONS]->(t)
WHERE c.spec_id IN t.source_specs
  AND c.chunk_type IN ['definition', 'general', 'requirement', 'overview']
WITH c, full_name, abbr,
  CASE
    WHEN c.section_title = abbr THEN 1.0
    WHEN full_name IS NOT NULL AND toLower(c.section_title) CONTAINS toLower(full_name) THEN 1.0
    WHEN toLower(c.section_title) CONTAINS toLower(abbr) THEN 0.95
    WHEN c.chunk_type = 'definition' THEN 0.85
    WHEN c.chunk_type = 'general'    THEN 0.82
    ELSE 0.78
  END AS score
RETURN c.chunk_id AS chunk_id, c.content AS content, c.spec_id AS spec_id,
       c.section_title AS section, score
ORDER BY score DESC LIMIT $top_k"""


def _tpl_a2(abbrevs: list[str], intent: Optional[str], **_) -> Optional[str]:
    if len(abbrevs) < 2:
        return None
    return f"""MATCH (t:Term) WHERE t.abbreviation IN {_abbr_list(abbrevs)}
WITH collect(t.full_name) AS names,
     reduce(acc = [], s IN collect(t.source_specs) | acc + s) AS all_specs,
     collect(t) AS terms
MATCH (c:Chunk)
WHERE c.spec_id IN all_specs
   OR ANY(n IN names WHERE n IS NOT NULL AND c.section_title CONTAINS n)
OPTIONAL MATCH (c)-[m:MENTIONS]->(tm:Term) WHERE tm IN terms
WITH c, names, m,
  CASE
    WHEN ANY(n IN names WHERE n IS NOT NULL AND c.section_title CONTAINS n) THEN 1.0
    WHEN m IS NOT NULL                 THEN 0.92
    WHEN c.chunk_type = 'definition'   THEN 0.85
    ELSE 0.6
  END AS score
RETURN c.chunk_id AS chunk_id, c.content AS content, c.spec_id AS spec_id,
       c.section_title AS section, score
ORDER BY score DESC LIMIT $top_k"""


def _tpl_f(abbrevs: list[str], **_) -> Optional[str]:
    if len(abbrevs) < 2:
        return None
    return f"""MATCH (t:Term) WHERE t.abbreviation IN {_abbr_list(abbrevs)}
WITH collect(t) AS terms
MATCH (c:Chunk)-[:HAS_STEP]->(st:Step)
WHERE ALL(term IN terms WHERE (st)-[:INVOLVES]->(term))
WITH c, count(DISTINCT st) AS n_steps
RETURN c.chunk_id AS chunk_id, c.content AS content, c.spec_id AS spec_id,
       c.section_title AS section, toFloat(n_steps) AS score
ORDER BY score DESC LIMIT $top_k"""


def _tpl_g(abbrevs: list[str], **_) -> Optional[str]:
    if len(abbrevs) < 2:
        return None
    return f"""MATCH (t:Term) WHERE t.abbreviation IN {_abbr_list(abbrevs)}
WITH collect(t) AS terms
MATCH (c:Chunk)
WHERE (c)-[:DESCRIBES_OPERATION]->(:ServiceOperation)
  AND ALL(term IN terms WHERE
        (c)-[:MENTIONS]->(term)
     OR (c)-[:DESCRIBES_OPERATION]->(:ServiceOperation)-[:PROVIDED_BY]->(term))
WITH c, terms,
     size([term IN terms WHERE (c)-[:DESCRIBES_OPERATION]->(:ServiceOperation)-[:PROVIDED_BY]->(term)]) AS n_via_op
RETURN c.chunk_id AS chunk_id, c.content AS content, c.spec_id AS spec_id,
       c.section_title AS section, toFloat(n_via_op) AS score
ORDER BY score DESC LIMIT $top_k"""


def _tpl_fg(abbrevs: list[str], **_) -> Optional[str]:
    """Union of F (step co-occurrence) and G (service-operation bridge) — used
    when the question is an NF↔NF interaction but F-vs-G is ambiguous. Both find
    the interaction from different angles (numbered steps vs SBI operations); the
    downstream internal rerank narrows to the relevant chunks, so taking both is
    strictly safer than betting on one.

    Score is a REAL per-chunk signal, not flat (2026-07-12, "AMF↔UDM registration"
    case): the flat 1.0 made the in-Cypher `LIMIT $top_k` cut chunks in arbitrary
    Neo4j order, so the gold chunk (ts_23_502_4.2.2.2.2 — General Registration,
    MENTIONS both AMF+UDM and carries Nudm_UECM_Registration/SDM_Get) fell outside
    the top-40 and never reached the rerank pool. Now:
      F branch: n_steps (how many steps co-involve all actors)
      G branch: n_via_op (actors visible as an SBI operation) + n_mention (actors
                the chunk MENTIONS directly). A chunk that MENTIONS BOTH actors
                AND describes their operations scores highest — exactly the
                General Registration chunk. `ORDER BY score DESC` before LIMIT so
                the widest, most-connected chunks survive the cut, not a random slice."""
    if len(abbrevs) < 2:
        return None
    lst = _abbr_list(abbrevs)
    # Single-statement form (NOT a UNION): a UNION's trailing ORDER BY/LIMIT only
    # bounds its LAST branch, so the gold chunk could still be cut from the F side.
    # Instead we gather the candidate set (chunks with a co-involving Step, OR
    # chunks that describe an operation and connect to all actors) then compute
    # BOTH signals per chunk and order the whole set once before LIMIT.
    return f"""MATCH (t:Term) WHERE t.abbreviation IN {lst}
WITH collect(t) AS terms
MATCH (c:Chunk)
WHERE (
        (c)-[:HAS_STEP]->(:Step)
        AND ALL(term IN terms WHERE (c)-[:HAS_STEP]->(:Step)-[:INVOLVES]->(term))
      )
   OR (
        (c)-[:DESCRIBES_OPERATION]->(:ServiceOperation)
        AND ALL(term IN terms WHERE
              (c)-[:MENTIONS]->(term)
           OR (c)-[:DESCRIBES_OPERATION]->(:ServiceOperation)-[:PROVIDED_BY]->(term))
      )
WITH c, terms,
     size([term IN terms WHERE (c)-[:HAS_STEP]->(:Step)-[:INVOLVES]->(term)]) AS n_step_actors,
     size([term IN terms WHERE (c)-[:DESCRIBES_OPERATION]->(:ServiceOperation)-[:PROVIDED_BY]->(term)]) AS n_via_op,
     size([term IN terms WHERE (c)-[:MENTIONS]->(term)]) AS n_mention
// Two dominant tiers, both "ALL asked actors participate in ONE chunk":
//   +100  step co-occurrence  (Pattern F: every actor in the chunk's steps)
//    +50  operation bridge     (Pattern G: every actor provides an SBI operation
//                               the chunk describes — the gold for G-003 sits
//                               here, ts_29_552 UE Mobility Analytics with both
//                               NWDAF+AF operations, and was drowned by mention-
//                               only ts_23_288 chunks before this tier existed)
// A chunk that merely MENTIONS all actors gets neither tier. Within a tier the
// finer counts order it. This keeps the G branch from flooding the pool AND the
// F branch from starving a genuine G-only question.
WITH c, terms, n_step_actors, n_via_op, n_mention,
     (CASE WHEN n_step_actors = size(terms) THEN 100 ELSE 0 END
      + CASE WHEN n_via_op = size(terms) THEN 50 ELSE 0 END
      + 2 * n_mention + n_via_op + n_step_actors) AS score
RETURN c.chunk_id AS chunk_id, c.content AS content, c.spec_id AS spec_id,
       c.section_title AS section, c.chunk_type AS chunk_type,
       toFloat(score) AS score,
       (n_step_actors = size(terms)) AS has_step_all_actors,
       (n_via_op >= 1) AS op_prefix_match
ORDER BY score DESC, n_step_actors DESC, n_via_op DESC, c.chunk_id ASC LIMIT $top_k"""


def _tpl_b(interface_id: Optional[str], **_) -> Optional[str]:
    if not interface_id:
        return None
    ident = _esc(interface_id)
    return f"""MATCH (c:Chunk)
WHERE c.section_title =~ '(?i).*\\\\b{ident}\\\\b.*'
RETURN c.chunk_id AS chunk_id, c.content AS content, c.spec_id AS spec_id,
       c.section_title AS section, 1.0 AS score
ORDER BY score DESC LIMIT $top_k"""


def _tpl_m(message_name: Optional[str], **_) -> Optional[str]:
    if not message_name:
        return None
    name = _esc(message_name)
    return f"""MATCH (c:Chunk)-[:DESCRIBES_MESSAGE]->(:Message {{name: '{name}'}})
RETURN c.chunk_id AS chunk_id, c.content AS content, c.spec_id AS spec_id,
       c.section_title AS section, 1.0 AS score
ORDER BY score DESC LIMIT $top_k"""


def _tpl_p(parameter_name: Optional[str], **_) -> Optional[str]:
    if not parameter_name:
        return None
    name = _esc(parameter_name)
    return f"""MATCH (p:Parameter {{name: '{name}'}})-[:DEFINED_IN_TABLE]->(c:Chunk)
RETURN DISTINCT c.chunk_id AS chunk_id, c.content AS content, c.spec_id AS spec_id,
       c.section_title AS section, 1.0 AS score
ORDER BY score DESC LIMIT $top_k"""


def _tpl_e(spec_ids: list[str], **_) -> Optional[str]:
    if not spec_ids:
        return None
    lst = "[" + ", ".join(f"'{_esc(s)}'" for s in spec_ids) + "]"
    return f"""MATCH (c:Chunk)-[r:REFERENCES_CHUNK {{is_external: true}}]->(t:Chunk)
WHERE c.spec_id IN {lst} OR t.spec_id IN {lst}
RETURN c.chunk_id AS chunk_id, c.content AS content, c.spec_id AS spec_id,
       c.section_title AS section, r.confidence AS score
UNION
MATCH (c:Chunk)-[r:REFERENCES_CHUNK {{is_external: true}}]->(t:Chunk)
WHERE c.spec_id IN {lst} OR t.spec_id IN {lst}
RETURN t.chunk_id AS chunk_id, t.content AS content, t.spec_id AS spec_id,
       t.section_title AS section, r.confidence AS score
ORDER BY score DESC LIMIT $top_k"""


def _tpl_d(phrases: list[str], **_) -> Optional[str]:
    phrases = [p for p in (phrases or []) if p and len(p) >= 3][:2]
    if not phrases:
        return None
    clauses = " OR ".join(
        f"c.section_title =~ '(?i).*\\\\b{_esc(p)}\\\\b.*'" for p in phrases
    )
    return f"""MATCH (c:Chunk)
WHERE {clauses}
RETURN c.chunk_id AS chunk_id, c.content AS content, c.spec_id AS spec_id,
       c.section_title AS section, 1.0 AS score
ORDER BY score DESC LIMIT $top_k"""


def _tpl_c(**_) -> str:
    return """MATCH (c:Chunk) WHERE false
RETURN c.chunk_id AS chunk_id, c.content AS content,
       c.spec_id AS spec_id, c.section_title AS section, 0.0 AS score
LIMIT $top_k"""


_BUILDERS = {
    "A1": _tpl_a1, "A2": _tpl_a2, "FUNC": _tpl_func,
    "F": _tpl_f, "G": _tpl_g, "FG": _tpl_fg,
    "B": _tpl_b, "M": _tpl_m, "P": _tpl_p, "E": _tpl_e, "D": _tpl_d, "C": _tpl_c,
}

VALID_PATTERNS = tuple(_BUILDERS.keys())


def build_cypher(pattern: str, slots: dict) -> Optional[str]:
    """Build the Cypher for `pattern` from `slots`. Returns None if the pattern
    is unknown or its required slots are absent (caller falls back to the LLM).
    `slots` keys: abbrevs (list[str]), intent (str), interface_id (str),
    message_name (str), parameter_name (str), spec_ids (list[str]),
    phrases (list[str])."""
    builder = _BUILDERS.get((pattern or "").strip().upper())
    if builder is None:
        return None
    # Fill every known slot with a default so builders can take named params
    # without KeyError when the caller omits an irrelevant slot.
    full = {
        "abbrevs": [], "intent": None, "interface_id": None,
        "message_name": None, "parameter_name": None, "spec_ids": [], "phrases": [],
        **(slots or {}),
    }
    return builder(**full)
