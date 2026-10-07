"""
LLM-driven Cypher generator. Given the live KG schema and the user's question
(plus resolved term context), asks an Ollama model to write a single read-only
Cypher query that returns chunk results.

Safety: rejects any query containing write keywords or multiple statements.
"""
import json
import os
import re
from collections.abc import Iterator
from typing import Optional

from llm import OllamaClient

from retrieval.cypher_templates import build_cypher, VALID_PATTERNS
from retrieval import wh_router

# Two-step Cypher generation (2026-07-12): LLM picks a pattern + extracts slot
# names in ONE short JSON call, then cypher_templates fills a fixed template
# deterministically — removes the run-to-run variance of the "pick AND write"
# single call (esp. Pattern G). Toggle off with CYPHER_2STEP=0 to use the legacy
# LLM-writes-the-whole-query path (generate_stream).
#
# DEFAULT FLIPPED TO 0 (2026-07-12): the 2-step WH-router + FG-merge template was
# measured as a REGRESSION on wh_kg_v3_100 — fixed R@5 0.282 vs the LLM-Cypher
# baseline 0.423 (see memory fg-template-vs-llm-cypher-regression). The FG-merge
# pattern floods the RRF pool with ~100 same-score chunks and evicts vector gold;
# the term-count-before-slot routing killed Pattern M/P. LLM-generated Cypher is
# more flexible (narrow query when the question needs one). We keep the G4 anchor,
# Part E rerank bias, and grounding-rule improvements (all independent of this
# path) and let the LLM write the Cypher. Set CYPHER_2STEP=1 to re-enable the
# template router.
CYPHER_2STEP = os.getenv("CYPHER_2STEP", "0") in ("1", "true", "True")

# Compact pattern-selection prompt. The model returns ONLY a small JSON object;
# it never writes Cypher, so it can't produce a broken/empty query. Slot names it
# extracts (message/parameter/interface) are copied verbatim from the question.
# Slot-extraction prompt (Tier 2 of the WH-router). The PATTERN is chosen
# deterministically from intent + #terms + spec_refs (see wh_router), so the LLM
# no longer picks a pattern — it only pulls the three named-entity slots that a
# term-less question might anchor on. Called ONLY when wh_router.needs_slot_extraction
# is true, so most term-anchored questions never hit the LLM here at all.
_SLOT_PROMPT = """Extract named anchors from a 3GPP question. Return ONLY a JSON object, no prose.

- message_name: an EXACT ALL-CAPS protocol message the question names (e.g. LOCATION REPORT, REGISTRATION REQUEST), else null.
- parameter_name: an EXACT parameter / information element the question names (e.g. "Transport Layer Address", "Procedure Code"), original case, else null.
- interface_id: an interface / reference-point identifier the question centres on (N1, N6, S1, S5, Xn, Uu), else null.

Return JSON: {"message_name": "<or null>", "parameter_name": "<or null>", "interface_id": "<or null>"}

# Question
{question}

JSON:"""


def _parse_selection(raw: str) -> dict:
    """Extract the {pattern, *_name, ...} object from the model's JSON reply.
    Tolerant of surrounding prose / code fences. Returns {} on failure."""
    text = (raw or "").strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


# Token cap for the Cypher-gen call. A valid Cypher (<~15 lines per prompt rule)
# fits well under this; the cap blocks degenerate generation (observed one query
# emit ~1.4 GB of tokens / ~9.6 min when many network-function terms resolved at
# once → huge context → model stuck looping). Override via env.
CYPHER_MAX_TOKENS = int(os.getenv("CYPHER_MAX_TOKENS", "1024"))


# Cypher write/admin keywords that must never appear in a generated query.
# Matched as whole words (case-insensitive) to allow them inside string literals
# that the validator will reject anyway via the semicolon/multi-statement check.
FORBIDDEN_KEYWORDS = (
    "CREATE", "MERGE", "DELETE", "DETACH", "SET", "REMOVE", "DROP",
    "LOAD CSV", "USING PERIODIC", "FOREACH",
    "CALL { ", "CALL{",  # block subqueries that may write
    "CALL DBMS.", "CALL APOC.CREATE", "CALL APOC.MERGE", "CALL APOC.LOAD",
    "CALL APOC.PERIODIC", "CALL APOC.CYPHER.RUN", "CALL APOC.DO.",
)

# Cypher fence patterns the LLM may emit; we strip them
FENCE_PATTERN = re.compile(r"^```(?:cypher|sql|neo4j)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)

# First Cypher head keyword in the LLM output — used to skip prose prefixes
# (e.g. "Answer: B\nJustification: ...") that some reasoning models leak in
# front of the actual query when the question contains MCQ choices.
CYPHER_HEAD_PATTERN = re.compile(r"\b(MATCH|OPTIONAL\s+MATCH|UNWIND|WITH)\b", re.IGNORECASE)

# Detects multiple-choice tail "\nA. ...", "\nA) ...", "\nB. ...", etc. used by
# TeleQnA-style benchmarks. We strip from the first such marker to keep only
# the question stem so the Cypher generator never sees "Begin your reply with
# Answer: <letter>" instructions meant for a different stage.
MCQ_PATTERN = re.compile(r"\n\s*[A-E][.)]\s")

# Matches "TS 23.501" / "TR38.300" / "ts 23-501" etc. (spec_refs comes back in
# this loose dotted form from term_first.py's _normalise_spec_refs/_scan_spec_refs
# — canonical KG spec_id is underscore-form "ts_23_501"). Used by
# _canonicalize_spec_ref() below to convert for Pattern E's literal spec lists.
_SPEC_REF_PATTERN = re.compile(r"\b(TS|TR)\s*(\d{2})[.\-_](\d{3})\b", re.IGNORECASE)


def _canonicalize_spec_ref(raw: str) -> Optional[str]:
    """'TS 23.501' -> 'ts_23_501' (canonical Chunk/Document.spec_id form). Returns
    None if the string doesn't match the expected "TS/TR NN.NNN" shape."""
    m = _SPEC_REF_PATTERN.search(raw)
    if not m:
        return None
    return f"ts_{m.group(2)}_{m.group(3)}"


class CypherValidationError(ValueError):
    pass


class LLMCypherGenerator:
    def __init__(self, llm: OllamaClient, model: str):
        self._llm = llm
        self._model = model

    def generate_stream(
        self,
        question: str,
        schema_text: str,
        intent: Optional[str] = None,
        resolved_terms: Optional[dict] = None,
        primary_term: Optional[str] = None,
        think: bool = True,
        vector_hints: Optional[list[dict]] = None,
        spec_refs: Optional[list[str]] = None,
        model: Optional[str] = None,
        num_ctx: Optional[int] = None,
        value_concepts: Optional[list[str]] = None,
    ) -> Iterator[dict]:
        """
        Stream Cypher generation token-by-token so the UI can show progress live.

        Yields events:
          {kind: 'prompt', prompt: str, model: str}                       — once at start
          {kind: 'thinking', token: str, accumulated: str}                — per thinking token (reasoning models)
          {kind: 'token', token: str, accumulated: str}                   — per Cypher token from LLM
          {kind: 'done', cypher: str, raw: str}                           — once at end (after validation)
          {kind: 'error', error: str, raw: str}                           — if validation fails
        """
        # Two-step path: pick pattern + slots (JSON), fill a deterministic template.
        # Falls through to the legacy single-call path only if the selection call
        # fails to yield a buildable (pattern, slots) — so it degrades gracefully.
        if CYPHER_2STEP:
            built = yield from self._select_and_build(
                question=question, intent=intent, resolved_terms=resolved_terms,
                spec_refs=spec_refs, vector_hints=vector_hints,
                model=model, num_ctx=num_ctx,
            )
            if built:
                return

        prompt = self._build_prompt(
            question=question,
            schema_text=schema_text,
            intent=intent,
            resolved_terms=resolved_terms,
            primary_term=primary_term,
            vector_hints=vector_hints,
            spec_refs=spec_refs,
            value_concepts=value_concepts,
        )
        # Per-call override so the orchestrator can keep both Cypher and answer
        # stages on the same model (no Ollama load/unload thrash). Falls back to
        # the construction-time default for callers that don't specify.
        chosen_model = model or self._model
        yield {"kind": "prompt", "prompt": prompt, "model": chosen_model}

        raw_parts: list[str] = []
        thinking_parts: list[str] = []
        try:
            # Reasoning models think first, then emit the Cypher. We forward both
            # phases so the UI can show progress during the (often long) thinking phase.
            # When think=False, the model jumps straight to the Cypher output.
            for ev in self._llm.generate_stream_full(
                prompt, model=chosen_model, think=think, num_predict=CYPHER_MAX_TOKENS,
                num_ctx=num_ctx,
            ):
                if ev["kind"] == "thinking":
                    thinking_parts.append(ev["token"])
                    yield {
                        "kind": "thinking",
                        "token": ev["token"],
                        "accumulated": "".join(thinking_parts),
                    }
                else:
                    raw_parts.append(ev["token"])
                    yield {
                        "kind": "token",
                        "token": ev["token"],
                        "accumulated": "".join(raw_parts),
                    }
        except Exception as e:
            yield {"kind": "error", "error": f"llm_error: {type(e).__name__}: {e}", "raw": "".join(raw_parts)}
            return

        raw = "".join(raw_parts)
        cypher = self._clean(raw)
        try:
            self._validate(cypher)
        except CypherValidationError as e:
            yield {"kind": "error", "error": f"validation: {e}", "raw": raw, "model": chosen_model}
            return
        yield {"kind": "done", "cypher": cypher, "raw": raw, "model": chosen_model}

    def _select_and_build(
        self,
        question: str,
        intent: Optional[str],
        resolved_terms: Optional[dict],
        spec_refs: Optional[list[str]],
        vector_hints: Optional[list[dict]],
        model: Optional[str],
        num_ctx: Optional[int],
    ):
        """Two-tier WH-router generation. Tier 1 classifies the WH-type from
        intent (deterministic). Tier 2 picks the concrete pattern from #terms /
        spec_refs — calling the LLM ONLY to extract named-entity slots when the
        WH-type's tree could resolve to a slot-bearing pattern (M/P/B/E). Yields
        the same event shapes as generate_stream. Returns True (via StopIteration
        value) on a valid query, False to fall back to the legacy single-call path."""
        abbrevs = list((resolved_terms or {}).keys())
        canonical_specs = []
        for raw_ref in (spec_refs or []):
            c = _canonicalize_spec_ref(raw_ref)
            if c and c not in canonical_specs:
                canonical_specs.append(c)
        chosen_model = model or self._model

        # Tier 1: WH-type from intent (no LLM).
        wh_type = wh_router.classify_wh(intent)
        n_terms = len(abbrevs)

        # Tier 2a: extract named-entity slots ONLY when the tree may need them.
        raw = ""
        sel: dict = {}
        if wh_router.needs_slot_extraction(wh_type, n_terms):
            prompt = _SLOT_PROMPT.replace(
                "{question}", LLMCypherGenerator._strip_mcq(question)
            )
            yield {"kind": "prompt", "prompt": prompt, "model": chosen_model}
            try:
                raw = self._llm.generate(
                    prompt, model=chosen_model, format="json", think=False,
                    num_predict=200, num_ctx=num_ctx,
                )
            except Exception as e:
                yield {"kind": "error", "error": f"select_error: {type(e).__name__}: {e}", "raw": ""}
                return False
            sel = _parse_selection(raw)

        slot_hints = {
            "message_name": sel.get("message_name") or None,
            "parameter_name": sel.get("parameter_name") or None,
            "interface_id": sel.get("interface_id") or None,
        }
        # Tier 2b: pick the concrete pattern id from the WH-type's tree (no LLM).
        pattern = wh_router.route(
            wh_type, n_terms, has_spec_ref=bool(canonical_specs), slots=slot_hints,
        )

        # Vector-hint phrases for Pattern D fallback (distinctive 2-3 word titles).
        phrases = [h.get("section") or h.get("section_title") or ""
                   for h in (vector_hints or [])][:2]
        slots = {
            "abbrevs": abbrevs,
            "intent": intent,
            "spec_ids": canonical_specs,
            "phrases": phrases,
            **slot_hints,
        }
        cypher = build_cypher(pattern, slots) if pattern in VALID_PATTERNS else None
        if cypher is None:
            # Route unusable (missing slot for the chosen pattern) — signal fallback.
            return False

        try:
            self._validate(cypher)
        except CypherValidationError:
            return False
        # Surface the WH-type + built Cypher so the UI trail shows the route taken.
        yield {"kind": "token", "token": f"-- WH:{wh_type} → {pattern}\n{cypher}",
               "accumulated": cypher}
        yield {"kind": "done", "cypher": cypher, "raw": raw, "model": chosen_model}
        return True

    # ----- internal -----

    @staticmethod
    def _strip_mcq(query: str) -> str:
        """Remove multiple-choice tail (A./B./C./...) and any trailing meta-instruction
        like 'Begin your reply with Answer: <letter>' from the question text. Returns
        the question stem only — keeps the Cypher generator from copying answer-stage
        instructions into its output.
        """
        if not query:
            return ""
        m = MCQ_PATTERN.search(query)
        return query[: m.start()].strip() if m else query.strip()

    @staticmethod
    def _build_prompt(
        question: str,
        schema_text: str,
        intent: Optional[str],
        resolved_terms: Optional[dict],
        primary_term: Optional[str],
        vector_hints: Optional[list[dict]] = None,
        spec_refs: Optional[list[str]] = None,
        value_concepts: Optional[list[str]] = None,
    ) -> str:
        # ---- DYNAMIC block (per-query — placed at end so STABLE prefix can be cached) ----
        question_stem = LLMCypherGenerator._strip_mcq(question)

        # Format resolved-term hints — full_name only. The Cypher patterns (A1/A2)
        # read `t.source_specs` straight off the Term node at query time, so the
        # LLM never needs the spec list as text — including it used to blow up
        # the prompt (UE resolves to ~250 source_specs, 5GC to ~65) to 15k+ chars,
        # which pushed reasoning models past their token budget mid-thinking and
        # produced empty/prose/multi-statement Cypher instead of a valid query.
        terms_block = ""
        if resolved_terms:
            # node_label is present when the entity is NOT a :Term (Message/ServiceOperation/
            # Concept/Parameter). Unless told so, the LLM defaults to
            # (t:Term {abbreviation:'<message name>'}) → always 0 rows.
            lines = []
            for abbr, info in resolved_terms.items():
                label = info.get("node_label")
                if label:
                    lines.append(
                        f"  - {abbr} → node (:{label} {{name: '{abbr}'}}) — "
                        f"MATCH this label by `name`, NOT (:Term {{abbreviation}})")
                else:
                    lines.append(f"  - {abbr} = {info.get('full_name', '?')}  (:Term)")
            terms_block = "Resolved terms (authoritative from KG):\n" + "\n".join(lines)
        elif primary_term:
            terms_block = f"Primary term hint: {primary_term}"
        else:
            terms_block = (
                "Resolved terms: (none) — use Pattern B if an interface identifier "
                "matches, else Pattern D (vector-seeded regex) when vector hints "
                "are present, else Pattern C."
            )

        intent_block = f"Intent classification: {intent}" if intent else "Intent classification: (unknown)"

        # Vector hints: top section_titles vector retrieval found. Title + spec_id only,
        # no content — keeps Cypher generator independent of vector scoring while still
        # giving it real anchors instead of forcing it to invent section names.
        if vector_hints:
            hint_lines = []
            for h in vector_hints[:5]:
                section = (h.get("section") or h.get("section_title") or "?").strip()
                spec = (h.get("spec_id") or "?").strip()
                if section and section != "?":
                    hint_lines.append(f"  - \"{section}\"  ({spec})")
            vector_hints_block = (
                "Vector hints (top section_titles vector found — use as anchors when relevant; do NOT invent):\n"
                + "\n".join(hint_lines)
            ) if hint_lines else "Vector hints: (none useful)"
        else:
            vector_hints_block = "Vector hints: (not provided)"

        # Explicit 3GPP spec numbers named in the question (e.g. "TS 23.501"),
        # pre-canonicalized to the KG's underscore spec_id form so the LLM can
        # copy them literally into Pattern E instead of doing the TS-NN.NNN ->
        # ts_NN_NNN conversion itself (error-prone for an LLM to do reliably).
        canonical_specs = []
        if spec_refs:
            for raw in spec_refs:
                c = _canonicalize_spec_ref(raw)
                if c and c not in canonical_specs:
                    canonical_specs.append(c)
        if canonical_specs:
            spec_refs_block = (
                "Explicit spec(s) named in the question (canonical spec_id form): "
                + ", ".join(f"'{s}'" for s in canonical_specs)
                + " — use Pattern E if the question asks how one of these relates to "
                  "/ references / cross-references another spec or topic."
            )
        else:
            spec_refs_block = "Explicit spec(s) named in the question: (none)"

        # Standardized-value concepts (Layer C) the question names verbatim —
        # detected upstream against the KG's Concept registry, so this block
        # (and Pattern V itself) only ever appears for questions where the
        # pattern is applicable. Kept OUT of the stable pattern menu: the menu
        # is cacheable and every extra always-on pattern measurably dilutes the
        # LLM's pattern choice on unrelated questions.
        if value_concepts:
            names = ", ".join(f"'{c}'" for c in value_concepts)
            value_block = f"""Standardized-value concept(s) named in the question (exist as Concept nodes): {names}
PATTERN V applies and OVERRIDES the menu above — the question asks about
standardized values (which value/type to use, what a value means). Emit:
  MATCH (co:Concept {{name: {list(value_concepts)[0]!r}}})-[:HAS_VALUE]->(v:StandardizedValue)-[:DEFINED_IN_TABLE]->(c:Chunk)
  WITH c, count(v) AS n_vals
  RETURN c.chunk_id AS chunk_id, c.content AS content, c.spec_id AS spec_id,
         c.section_title AS section, n_vals AS score
  ORDER BY score DESC LIMIT $top_k
(Copy the concept name EXACTLY as listed above.)"""
        else:
            value_block = ""

        # ---- STABLE block (role + schema + rules + examples — cacheable across queries) ----
        return f"""You write a single Cypher query for a Neo4j knowledge graph of 3GPP technical specifications.

# Purpose
The graph branch BACKS UP a parallel vector branch via Reciprocal Rank Fusion.
Your job is to retrieve Chunk nodes anchored on canonical KG structure (Term
abbreviations, section_title regexes). Do NOT try to answer the user's
question. Do NOT include prose, explanations, or letter answers.

# Real KG schema (verified against live database)

Node properties:
- Term:           abbreviation, full_name, primary_spec, source_specs, term_type
- Chunk:          chunk_id, spec_id, section_id, section_title, content,
                  chunk_type, subject, key_terms, is_parent_section
- Document:       spec_id, version, title, total_chunks
- Subject:        name, priority, description (generic categories — NOT term-specific)
- Step:           step_id, chunk_id, order, text (ONE numbered step, e.g. "1", "3a",
                  inside a chunk_type='procedure' Chunk — see Pattern F)
- ServiceOperation: name, nf_prefix, service, operation (a 3GPP SBI operation,
                  e.g. name='Nudm_SDM_Get' — see Pattern G)
- Message:        name (ALL-CAPS NAS/RRC/application-protocol message, global-unique
                  key, e.g. name='LOCATION REPORT' — see Pattern M)
- Parameter:      name, description (a table field / information element, e.g.
                  name='Transport Layer Address' — see Pattern P)

Relationships (verified counts after rebuild):
- (Chunk)-[:HAS_SUBJECT]->(Subject)        # ~197k — Subject is a GENERIC category, NOT a Term. DO NOT use to find chunks for a term.
- (Chunk)-[:REFERENCES_SPEC]->(Document)   # ~165k — chunk-to-doc link, coarse (which specs a chunk cites, no specific target chunk). NOT used by any pattern below — REFERENCES_CHUNK is strictly more precise for cross-spec questions.
- (Chunk)-[:REFERENCES_CHUNK]->(Chunk)     # internal section refs + cross-spec section refs. USE THIS for cross-spec citation questions (Pattern E) — it's pre-resolved to the specific cited chunk, not just the document.
                                           #   Edge property `is_external`: false=same-spec, true=cross-spec (~46k external edges, confidence>=0.7).
                                           #   Other properties: ref_type ('clause'), ref_id, confidence.
- (Document)-[:CONTAINS]->(Chunk)          # ~195k — every Chunk has incoming CONTAINS from its Document.
- (Term)-[:DEFINED_IN]->(Document)         # ~74k — Term defined in Document.
- (Chunk)-[:MENTIONS]->(Term)              # ~539k — chunk-level term mention (key_terms ∩ Term, df-filtered).
                                           #   Use as a PRECISION BOOST inside Pattern A (OPTIONAL MATCH + score m IS NOT NULL higher),
                                           #   NOT as the sole WHERE anchor (that cuts graph recall). Edge property `df`.
- (Chunk)-[:PARENT_SECTION]->(Chunk)       # ~44k — section hierarchy (child → nearest parent).
- (Chunk)-[:HAS_STEP]->(Step)              # ~20k — Step = 1 numbered step inside a chunk_type='procedure' Chunk.
- (Step)-[:INVOLVES]->(Term)               # ~39k — Term actually mentioned INSIDE that specific step's text
                                           #   (already df-filtered — same curated set as MENTIONS). Use for
                                           #   Pattern F: finds the SPECIFIC step where 2+ actors co-occur,
                                           #   not just any chunk mentioning both anywhere.
- (Chunk)-[:DESCRIBES_OPERATION]->(ServiceOperation)  # ~6k — chunk describes/invokes this SBI operation.
- (ServiceOperation)-[:PROVIDED_BY]->(Term)           # ~841 — the network function that PROVIDES the operation
                                           #   (the NF is encoded in the operation name's prefix, e.g. Nudm_* → UDM).
                                           #   Pattern G uses this to reach chunks where an NF appears ONLY as
                                           #   `NB_Service_Op` and never as the bare abbreviation in key_terms.
- (Chunk)-[:DESCRIBES_MESSAGE]->(Message)             # ~19k — chunk describes this ALL-CAPS message. Pattern M.
- (Parameter)-[:DEFINED_IN_TABLE]->(Chunk)            # ~78k — parameter/IE defined in this chunk's table. Pattern P.

# KG quirks — CRITICAL
- To find chunks for a term, anchor recall on `c.spec_id IN t.source_specs`, then
  BOOST precision with the direct edge via `OPTIONAL MATCH (c)-[m:MENTIONS]->(t)` and
  score `m IS NOT NULL` higher (chunk actually mentions the term). See Pattern A1/A2.
  Do NOT anchor the WHERE solely on `(c)-[:MENTIONS]->(t)` — it cuts graph recall.
- `Term.full_name` (NOT `Term.name` — that property does not exist).
- `Term.primary_spec` is **document-level** (e.g. `'ts_23_501'`), the single spec
  judged to define the term. Prefer `source_specs` for recall; `primary_spec`
  holds one value only and is not a section id.
- `Term.source_specs` is the list of section-level spec_ids where the term
  appears. Use `IN` against `c.spec_id`.
- spec_ids exist in DUPLICATE FORMATS (`ts_29.500` AND `ts_29_500`) — same
  content, different nodes. Live with it; do not try to dedup at query time.
- DO NOT use `chunk.content CONTAINS ...` — duplicates vector search, full
  scan, low precision.
- 3GPP Release / version (Rel-17, Rel-18, R18) is NOT reliably populated on
  `Document.version` — DO NOT filter on `d.version` or other version fields.
  If the user asks about a release, ignore the release filter and rely on
  spec_id structure.
- Property access on anonymous node literals is INVALID Cypher
  (`(:Document {{spec_id: 'x'}}).version` parses as a syntax error). Always
  bind a variable: `MATCH (d:Document {{spec_id: 'x'}}) RETURN d.version`.

# Performance rules — query MUST run <500ms (197k Chunks, 18k Terms)

1. **Indexed properties** (RANGE index): `Term.abbreviation`, `Document.spec_id`,
   `Chunk.spec_id`, `Chunk.chunk_id`, `Chunk.chunk_type`. Match them with `=`
   (NOT `toLower(...)`).
2. **Never wrap an indexed property in `toLower()` in WHERE** — bypasses the
   index. `Term.abbreviation` is canonical upper case already.
3. **Pipeline with `WITH` between MATCH clauses** to avoid cartesian products.
4. **Carry `full_name` forward** with `WITH t, t.full_name AS full_name LIMIT 1`
   so subsequent CASE / WHERE clauses can reference it without re-fetching.

# Intent → chunk_type mapping (verified empirically against KG)
- definition / what_is / abbreviation / network_function:
    chunk_type IN ['definition', 'abbreviation']
- procedure / how_does:
    chunk_type IN ['procedure', 'requirement']
- general / overview:
    chunk_type IN ['definition', 'general']
NOTE: chunk_type='interface' in source data is mislabeled — DO NOT filter on it.
For interface/api/signaling questions involving an identifier (N1, N6, S1...),
use Pattern B below.

# Ten retrieval patterns — pick exactly ONE based on the question

## Pattern A1 — Single term anchor (use when resolved_terms has exactly one entry)

  MATCH (t:Term {{abbreviation: '<TERM_UPPER>'}})
  WITH t, t.full_name AS full_name LIMIT 1
  MATCH (c:Chunk)
  WHERE (c.spec_id IN t.source_specs AND c.chunk_type IN [<INTENT_TYPES>])
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
  ORDER BY score DESC LIMIT $top_k

## Pattern A2 — Multi-term anchor (use when resolved_terms has ≥2 entries)
NEVER write `MATCH (t1) OR (t2)` — that is invalid Cypher. Collect terms in
ONE pass, then traverse Chunks once. Carry `names` and `all_specs` forward.

  MATCH (t:Term) WHERE t.abbreviation IN ['<ABBR1>', '<ABBR2>']
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
  ORDER BY score DESC LIMIT $top_k

## Pattern F — Procedure step co-occurrence (use when intent is 'procedure' or
## 'how_does' AND resolved_terms has ≥2 entries — MORE PRECISE than Pattern A2
## for "how does X interact with Y" questions: finds the SPECIFIC step where
## both actors appear TOGETHER, not just any chunk that mentions both anywhere)

  MATCH (t:Term) WHERE t.abbreviation IN ['<ABBR1>', '<ABBR2>']
  WITH collect(t) AS terms
  MATCH (c:Chunk)-[:HAS_STEP]->(st:Step)
  WHERE ALL(term IN terms WHERE (st)-[:INVOLVES]->(term))
  WITH c, count(DISTINCT st) AS n_steps
  RETURN c.chunk_id AS chunk_id, c.content AS content, c.spec_id AS spec_id,
         c.section_title AS section, toFloat(n_steps) AS score
  ORDER BY score DESC LIMIT $top_k

Rules for Pattern F:
- Only use with ≥2 resolved terms — step co-occurrence is meaningless for a
  single actor (use Pattern A1 instead for exactly 1 resolved term).
- `ALL(...)` requires every resolved term's Step-INVOLVES edge, not just one —
  do NOT relax to ANY, that degrades back to chunk-level (Pattern A2) behavior.
- The score is `n_steps` = how many of the chunk's steps co-involve ALL the
  actors. A chunk where the two actors interact across several steps is a
  stronger answer than one where they co-occur in a single incidental step, so
  this ranks the genuinely-interactive procedures to the top (do NOT replace it
  with a constant like `1.0` — a constant makes ORDER BY/LIMIT cut arbitrarily).
- Zero rows is an ACCEPTABLE outcome (Step/INVOLVES coverage is intentionally
  precision-first, not every procedure chunk is split into steps) — the
  parallel vector branch still carries the question via RRF. Do not add a
  fallback OR-clause to Chunk-level matching inside this same query; if
  uncertain whether Step coverage exists, prefer Pattern A2 instead.

## Pattern G — Service-operation bridge (use when intent is 'procedure',
## 'how_does', or 'relationship' AND resolved_terms has ≥2 network functions —
## RECOVERS chunks where a network function appears ONLY as its Service-Based
## Interface operation name (Nudm_..., Nausf_..., Namf_...) rather than as the
## bare abbreviation. Those chunks are INVISIBLE to Pattern A2/F because the
## bare token (e.g. "UDM") is not in key_terms — but the ServiceOperation node
## bridges chunk → operation → PROVIDED_BY → the network function Term.)

  MATCH (t:Term) WHERE t.abbreviation IN ['<ABBR1>', '<ABBR2>']
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
  ORDER BY score DESC LIMIT $top_k

Rules for Pattern G:
- Each resolved term is matched via MENTIONS OR via an operation it PROVIDES —
  a chunk qualifies only when EVERY resolved term is present through one of
  those two paths (`ALL(...)`). This is what recovers the mixed case where
  actor A is a bare mention but actor B only shows up as `NB_Service_Op`.
- The score is `n_via_op` = how many of the resolved actors appear as an actual
  SBI operation in the chunk (not just a bare mention). A chunk where BOTH NFs
  show up as service operations is describing their SBI interaction directly, so
  it outranks a chunk where one is only mentioned in passing (do NOT use a
  constant like `0.9` — that makes ORDER BY/LIMIT cut arbitrarily and drops gold).
- Keep the leading `(c)-[:DESCRIBES_OPERATION]->(:ServiceOperation)` guard —
  it restricts to chunks that actually describe SBI flows (precision), so the
  MENTIONS branch can't drag in generic chunks that merely name both terms.
- Prefer Pattern F FIRST for 'procedure'/'how_does' with ≥2 terms (step-level
  is more precise). Use Pattern G when the interaction is expressed through
  service-operation calls rather than numbered steps, or as the natural choice
  for intent 'relationship' with ≥2 network functions.
- Do NOT combine the `Nxxx_Yyy_Zzz` string into a `content CONTAINS` — that is
  forbidden (rule 8). The PROVIDED_BY edge is the pre-resolved, indexed path.

## Pattern B — Section-identifier word-boundary regex
Use when the question references an interface/reference-point identifier
matching `^[A-Z]\\d+[a-z]*$` (N1, N2, N6, S1, S5, Xn, X2) or known
identifiers (Uu). These are NOT stored as standalone Term abbreviations.

  MATCH (c:Chunk)
  WHERE c.section_title =~ '(?i).*\\\\b<IDENT>\\\\b.*'
  RETURN c.chunk_id AS chunk_id, c.content AS content, c.spec_id AS spec_id,
         c.section_title AS section, 1.0 AS score
  ORDER BY score DESC LIMIT $top_k

Doubled backslashes are required to escape `\\b` inside a Cypher string.
For multiple identifiers, OR them: `c.section_title =~ '(?i).*\\\\b(N6|S1)\\\\b.*'`.

## Pattern D — Vector-seeded section_title regex (use when NO Term anchor and NO
## interface identifier, BUT the Vector hints block lists real section_titles)
This turns the "vector-only" branch into "vector + graph-expand around the best
vector result". Pick the 1–2 MOST DISTINCTIVE multi-word phrases (2–3 content
words, skip stop-words / generic words like "the", "of", "requirements",
"general", "overview", "service") from the TOP vector-hint section_titles, and
match other chunks whose section_title contains those phrases. This recovers
sibling/related sections the pure vector top-k missed.

  MATCH (c:Chunk)
  WHERE c.section_title =~ '(?i).*\\\\b<PHRASE1>\\\\b.*'
     OR c.section_title =~ '(?i).*\\\\b<PHRASE2>\\\\b.*'
  RETURN c.chunk_id AS chunk_id, c.content AS content, c.spec_id AS spec_id,
         c.section_title AS section, 1.0 AS score
  ORDER BY score DESC LIMIT $top_k

Rules for Pattern D:
- Doubled backslashes are required to escape `\\b` inside a Cypher string.
- A PHRASE may contain a literal space (`\\\\bUser Configuration\\\\b`) — that is fine.
- Use ONLY phrases that actually appear in the Vector hints block. Do NOT invent
  section names. If no vector hint carries a distinctive phrase, fall back to
  Pattern C instead.
- Keep to ≤2 phrases so the regex stays selective (avoid over-matching).

## Pattern E — Cross-spec reference traversal (use when the "Explicit spec(s)
## named in the question" block above lists spec_id(s) AND the question asks
## how one spec relates to / references / cross-references another spec,
## procedure, or parameter defined elsewhere)
Use the pre-resolved `REFERENCES_CHUNK {{is_external: true}}` edge — it already
points from a chunk that cites another spec straight to the SPECIFIC chunk
being cited (not just the document), with a `confidence` score. This answers
"how does X in spec A relate to spec B" by finding the actual citing chunk(s)
and the actual cited chunk(s) on both ends, not just any chunk in each spec.

NOTE — the spec-pair filter alone is not perfectly selective (two large specs
can have many citation edges between them), and `confidence` measures how
sure the citation EXTRACTION was, not how relevant a specific citation is to
this question. Keep the query to EXACTLY the shape below — do not add extra
filter clauses beyond the spec-pair WHERE.

  MATCH (c:Chunk)-[r:REFERENCES_CHUNK {{is_external: true}}]->(t:Chunk)
  WHERE c.spec_id IN [<SPEC_LIST>] OR t.spec_id IN [<SPEC_LIST>]
  RETURN c.chunk_id AS chunk_id, c.content AS content, c.spec_id AS spec_id,
         c.section_title AS section, r.confidence AS score
  UNION
  MATCH (c:Chunk)-[r:REFERENCES_CHUNK {{is_external: true}}]->(t:Chunk)
  WHERE c.spec_id IN [<SPEC_LIST>] OR t.spec_id IN [<SPEC_LIST>]
  RETURN t.chunk_id AS chunk_id, t.content AS content, t.spec_id AS spec_id,
         t.section_title AS section, r.confidence AS score
  ORDER BY score DESC LIMIT $top_k

Rules for Pattern E:
- `<SPEC_LIST>` = the exact canonical spec_id(s) given in the "Explicit spec(s)
  named in the question" block (e.g. `['ts_23_501', 'ts_33_401']`). Do NOT
  invent or guess a spec_id yourself — only use the ones given to you there.
- If only ONE spec_id is given (question names one spec and asks what it
  references, or what references it), the `OR` clause still works correctly
  (matches either direction) — no need for two specs to use this pattern.
- If the "Explicit spec(s)" block says "(none)", this pattern is NOT
  available — fall back to Pattern D or C.
- This returns BOTH ends of the citation (the citing chunk and the cited
  chunk) via UNION so RRF/rerank sees the complete cross-reference pair, not
  just one side of it.
- Copy the query shape above VERBATIM aside from `<SPEC_LIST>` — do not add a
  topic/keyword filter clause. (A stricter topic-filtered variant was tried
  and measured WORSE on both retrieval and answer quality — it made the
  query longer and increased how often the model broke Cypher-only output
  format on complex multi-spec questions. Keep this pattern short.)

## Pattern M — Message cross-spec (use when the question names a specific 3GPP
## message — an ALL-CAPS phrase like "LOCATION REPORT", "REGISTRATION REQUEST",
## "WRITE-REPLACE WARNING REQUEST" — and asks which specs/protocols describe it
## or how it differs across them)
Copy the message name VERBATIM from the question (it is an exact ALL-CAPS node
key). Retrieve every chunk that describes it; the downstream reranker picks the
most relevant per-spec descriptions.

  MATCH (c:Chunk)-[:DESCRIBES_MESSAGE]->(:Message {{name: '<MESSAGE NAME>'}})
  RETURN c.chunk_id AS chunk_id, c.content AS content, c.spec_id AS spec_id,
         c.section_title AS section, 1.0 AS score
  ORDER BY score DESC LIMIT $top_k

Rules for Pattern M:
- `<MESSAGE NAME>` must be the EXACT ALL-CAPS message phrase from the question
  (e.g. 'LOCATION REPORT'). Do NOT lowercase it, do NOT paraphrase.
- Message.name is a global-unique node key — this is an indexed exact match.
- If the question names NO ALL-CAPS message phrase, this pattern does not apply.

## Pattern P — Parameter / Information-Element cross-spec (use when the question
## names a specific parameter or information element — often title-case like
## "Transport Layer Address", "Procedure Code", "Validity Time" — and asks where
## it is defined or how it differs across specifications)
Copy the parameter name VERBATIM from the question.

  MATCH (p:Parameter {{name: '<PARAMETER NAME>'}})-[:DEFINED_IN_TABLE]->(c:Chunk)
  RETURN DISTINCT c.chunk_id AS chunk_id, c.content AS content, c.spec_id AS spec_id,
         c.section_title AS section, 1.0 AS score
  ORDER BY score DESC LIMIT $top_k

Rules for Pattern P:
- `<PARAMETER NAME>` must be the EXACT name as written in the question, in its
  original case (Parameter.name is stored case-sensitively, e.g. 'Transport
  Layer Address'). Do NOT uppercase or paraphrase.
- Use this — NOT Pattern A — when the question is about a table field / IE / RRC
  or NAS parameter defined across specs, not about a Term abbreviation.

## Pattern C — No structural anchor (sentinel empty)
If NONE of the above fit (no Term, no interface identifier, AND no usable
vector-hint phrase — e.g. a purely conceptual question like "key benefit of
network slicing" with no vector hints), emit EXACTLY:

  MATCH (c:Chunk) WHERE false
  RETURN c.chunk_id AS chunk_id, c.content AS content,
         c.spec_id AS spec_id, c.section_title AS section, 0.0 AS score
  LIMIT $top_k

This signals "vector search should carry this query" and avoids polluting
RRF with hallucinated matches. NEVER match on Subject nodes.

# Rules — failure to follow ANY rule will be rejected
1. READ-ONLY only. Allowed keywords: MATCH, OPTIONAL MATCH, WHERE, WITH, RETURN, ORDER BY, LIMIT, SKIP, UNION, UNWIND, CASE, AS, DISTINCT.
2. NEVER use: CREATE, MERGE, DELETE, DETACH, SET, REMOVE, DROP, LOAD CSV, FOREACH, CALL with write side-effects.
3. EXACTLY ONE statement. No semicolons.
4. Return EXACTLY these columns (alias if needed):
     chunk_id, content, spec_id, section, score
5. Use the parameter `$top_k` for the LIMIT.
6. Use only the labels and relationship types listed in the schema above.
7. Keep the query compact (under ~15 lines).
8. FORBIDDEN: `chunk.content CONTAINS ...`. Filter on `section_title` only.
9. FORBIDDEN: filtering on `Document.version` / Release / `d.version`.
10. The query MUST end with `RETURN ...`. Never end with bare `WITH`.
11. Pattern selection priority:
    a. If the question names a specific ALL-CAPS message phrase (e.g. LOCATION
       REPORT) and asks which specs describe it / how it differs, use Pattern M.
    b. Else if the question names a specific parameter / information element
       (e.g. "Transport Layer Address") and asks where it is defined / how it
       differs across specs, use Pattern P.
    c. Else if the question explicitly asks how one named spec relates to /
       references / cross-references another spec, procedure, or parameter
       (and the "Explicit spec(s) named in the question" block is non-empty),
       use Pattern E — it answers a different question (cross-spec citation)
       than A1/A2/F (single-spec term lookup) even when a Term also resolved.
    d. Else if Intent classification is 'procedure' or 'how_does' AND
       resolved_terms has ≥2 entries, use Pattern F (step-level co-occurrence
       — more precise than A2 for "how does X interact with Y" questions).
    e. Else if Intent classification is 'relationship' with ≥2 network-function
       terms, OR the interaction is described through service-operation calls
       (Nxxx_Yyy_Zzz) rather than numbered steps, use Pattern G (service-
       operation bridge — recovers chunks where an NF appears only as its SBI
       operation name and is invisible to A2/F).
    f. Else if resolved_terms has exactly one entry, use Pattern A1; if ≥2,
       use Pattern A2.
    g. Else if an interface/reference-point identifier matches, use Pattern B.
    h. Else if the Vector hints block lists a distinctive multi-word phrase,
       use Pattern D.
    i. Else use Pattern C.

# Output format — STRICT
Your output MUST start with one of these tokens:
  MATCH | OPTIONAL MATCH | WITH | UNWIND
The very first character cannot be a letter A–E, the word "Answer", "Cypher",
"Query", or any prose. The user's question may include "Begin your reply with
`Answer: <letter>`" or A./B./C./D./E. choices — IGNORE all of that. You are
NOT answering the question. You are writing a Neo4j Cypher query that
retrieves chunks helpful for someone else to answer it. Do NOT include any
text outside the Cypher query. No prose. No comments. No markdown fences.

# ----- DYNAMIC context (per-query) -----
{intent_block}

{terms_block}

{vector_hints_block}

{spec_refs_block}

{value_block}

# User question (stem only — choices and meta-instructions stripped)
{question_stem}
"""

    @staticmethod
    def _clean(raw: str) -> str:
        text = (raw or "").strip()
        # Strip fenced blocks if present
        text = FENCE_PATTERN.sub("", text).strip()
        # Skip everything before the first Cypher head keyword. Catches prose
        # leaks like "Answer: B\nJustification: ...\nMATCH (n) RETURN n" —
        # which the previous prefix-stripper missed once the leak grew past
        # a single label word.
        m = CYPHER_HEAD_PATTERN.search(text)
        if m:
            text = text[m.start():]
        # Drop trailing semicolons
        return text.rstrip(";").strip()

    @staticmethod
    def _validate(cypher: str) -> None:
        if not cypher:
            raise CypherValidationError("LLM produced empty Cypher")
        # Reject multi-statement
        if ";" in cypher:
            raise CypherValidationError("Multiple Cypher statements are not allowed")
        # Reject concatenated statements that DON'T use ';' — observed live: with
        # 7 patterns to choose from, the model sometimes emits several complete
        # "MATCH ... RETURN ..." candidates back-to-back instead of picking one
        # (no semicolon, so the check above misses it; Neo4j itself rejects it at
        # runtime with "RETURN can only be used at the end of the query", but
        # catching it here gives a clearer error and fails faster). A legitimate
        # single statement has at most one RETURN per UNION segment (Pattern E
        # is the only pattern with 2 RETURNs, joined by UNION).
        for segment in re.split(r"\bUNION\b", cypher, flags=re.IGNORECASE):
            if segment.upper().count("RETURN") > 1:
                raise CypherValidationError(
                    "Multiple Cypher statements concatenated without UNION"
                )
        upper = cypher.upper()
        for kw in FORBIDDEN_KEYWORDS:
            if kw in upper:
                raise CypherValidationError(f"Forbidden keyword/clause in generated Cypher: {kw!r}")
        # Must contain at least one MATCH or UNWIND or RETURN — otherwise it's not a useful read query
        if not any(k in upper for k in ("MATCH", "UNWIND", "RETURN")):
            raise CypherValidationError("Generated Cypher does not contain MATCH/UNWIND/RETURN")
        # Must end with a RETURN clause (possibly followed by ORDER BY / LIMIT / SKIP).
        # A query ending on bare WITH is a Neo4j syntax error ("Query cannot conclude with WITH").
        if "RETURN" not in upper:
            raise CypherValidationError("Generated Cypher must contain a RETURN clause")
        # Find the last RETURN; only ORDER BY / LIMIT / SKIP / DESC / ASC / commas / identifiers / params allowed after.
        last_return_idx = upper.rfind("RETURN")
        last_with_idx = upper.rfind("WITH")
        if last_with_idx > last_return_idx:
            raise CypherValidationError("Generated Cypher cannot end with WITH — must end with RETURN ...")
