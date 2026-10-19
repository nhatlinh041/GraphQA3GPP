"""
Prompt builder + JSON action parser for the Adaptive ReAct retrieval agent.

The planner LLM emits a strict JSON object describing ONE next action per
iteration. Free-form prose is rejected — we re-prompt once, then fall back to
`finish` to keep the loop bounded.
"""
import json
import re
from dataclasses import dataclass
from typing import Any, Optional


# Tool names the planner may emit. Anything else is rejected.
ALLOWED_TOOLS = ("cypher_query", "expand_term", "vector_search", "inspect_chunk", "finish")

# Strip ```json ... ``` style fences if the model adds them despite instructions.
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)

# Greedy-but-balanced JSON object extractor — finds the FIRST {...} block.
# Used when the model wraps the JSON in stray prose.
_FIRST_OBJ = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class PlannerAction:
    thought: str
    tool: str
    args: dict
    remaining_gaps: list[str]
    raw: str


class PlannerParseError(ValueError):
    pass


# Truncate a long string for inclusion in the prompt history block.
def _truncate(s: str, n: int = 140) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


# Format the resolved-terms block (authoritative full names from KG).
def _resolved_terms_block(resolved_terms: Optional[dict]) -> str:
    if not resolved_terms:
        return "(none resolved)"
    lines = []
    for abbr, info in resolved_terms.items():
        full = info.get("full_name", "?")
        specs = info.get("specs") or []
        spec_str = f"  source_specs={specs}" if specs else ""
        lines.append(f"  - {abbr} = {full}{spec_str}")
    return "\n".join(lines)


# Compact preview of chunks already collected so the planner can see scope + dedup.
def _chunks_block(chunks: list[dict], max_show: int = 8) -> str:
    if not chunks:
        return "(no chunks yet)"
    lines = []
    for i, c in enumerate(chunks[:max_show], start=1):
        spec = c.get("spec_id", "?")
        section = c.get("section_title") or c.get("section") or "?"
        preview = _truncate(c.get("content", ""), 120)
        lines.append(f"  {i}. [{spec} §{section}] {preview}")
    if len(chunks) > max_show:
        lines.append(f"  … and {len(chunks) - max_show} more")
    return "\n".join(lines)


# Compact history of past actions (tool + key args + observation count).
def _history_block(history: list[dict]) -> str:
    if not history:
        return "(first iteration — no history)"
    lines = []
    for h in history:
        tool = h.get("tool", "?")
        args_brief = _truncate(json.dumps(h.get("args") or {}, ensure_ascii=False), 100)
        obs = h.get("observation_summary", "")
        lines.append(f"  iter {h.get('iter')}: {tool}({args_brief}) → {obs}")
    return "\n".join(lines)


# Main planner prompt — includes live KG schema reminders + tool catalog.
def build_planner_prompt(
    *,
    question: str,
    intent: str,
    resolved_terms: Optional[dict],
    seeds: list[str],
    chunks: list[dict],
    gaps: list[str],
    history: list[dict],
    iter_idx: int,
    max_iter: int,
    last_error: Optional[str] = None,
) -> str:
    error_block = ""
    if last_error:
        error_block = (
            f"\n# Last action failed\n"
            f"Reason: {last_error}\n"
            f"Pick a DIFFERENT tool or fix the args. Do NOT repeat the same call.\n"
        )

    return f"""You are a retrieval planner for a 3GPP knowledge graph.
Your job: pick ONE next action to fill the remaining information gaps for the user's question.
You decide when to stop — call `finish` as soon as the chunks already collected are enough to answer.

# User question
{question}

# Intent
{intent}

# Seeds (from query parser)
{seeds or '(none)'}

# Resolved terms (authoritative from KG)
{_resolved_terms_block(resolved_terms)}

# Live KG schema — TRUST THIS over any other knowledge
Node properties:
- Term:    abbreviation, full_name, source_specs (list[str]), primary_spec, term_type
- Chunk:   chunk_id, spec_id, section_id, section_title, chunk_type, content, is_parent_section (bool)
- Document: spec_id, version, title, total_chunks
- Subject (do NOT use Subject for term lookup — it's a generic category)

Edges that are populated and SAFE to use (counts after rebuild with new schema):
- (Chunk)-[:HAS_SUBJECT]->(Subject)        ~197k  (only 5 generic subjects — low signal)
- (Chunk)-[:REFERENCES_SPEC]->(Document)   ~165k  (chunk-to-doc, USE for cross-spec at doc level)
- (Chunk)-[:REFERENCES_CHUNK]->(Chunk)     internal + cross-spec section refs.
                                           Edge prop `is_external` (bool): false=same-spec, true=cross-spec.
                                           Other props: ref_type ('clause'), ref_id, confidence.
                                           USE for section-precise cross-ref. Variable-length
                                           [:REFERENCES_CHUNK*1..3] traverses both internal + external uniformly.
- (Document)-[:CONTAINS]->(Chunk)          ~195k  (every Chunk has incoming CONTAINS from its Document).
                                           USE for doc→chunks traversal (e.g. "list all chunks in TS 23.501").
- (Term)-[:DEFINED_IN]->(Document)         ~5k    (Term defined in Document). SAFE but the gold pattern for
                                           "chunks mentioning a term" is `c.spec_id IN t.source_specs`
                                           (property lookup, faster than traversal). Use DEFINED_IN only
                                           for "list all terms defined in spec X" type queries.
- (Chunk)-[:PARENT_SECTION]->(Chunk)       ~100k  (section hierarchy, child→nearest parent).
                                           Property `Chunk.is_parent_section` (bool) marks chunks with children.
                                           USE for "overview/children of section X" via [:PARENT_SECTION*] traversal.

# Gold pattern (use this in cypher_query when looking up term-related chunks)
There is NO direct (Chunk)-(Term) edge. Use `c.spec_id IN t.source_specs`:

  MATCH (t:Term {{abbreviation: 'NWDAF'}})
  WITH t, t.full_name AS full_name LIMIT 1
  MATCH (c:Chunk)
  WHERE c.spec_id IN t.source_specs
     OR (full_name IS NOT NULL AND c.section_title CONTAINS full_name)
  RETURN c.chunk_id AS chunk_id, c.content AS content, c.spec_id AS spec_id,
         c.section_title AS section, 0.9 AS score
  ORDER BY score DESC LIMIT $top_k

# Current state
- Iteration: {iter_idx} of {max_iter}
- Chunks already collected ({len(chunks)}):
{_chunks_block(chunks)}

- Open gaps:
{chr(10).join('  - ' + g for g in gaps) if gaps else '  (none — you can finish)'}

- Action history:
{_history_block(history)}
{error_block}
# Available tools (pick exactly ONE)
1. vector_search — semantic similarity over chunk embeddings. Best FIRST CALL when
   you have no prior chunks (the question may map to chunks via natural language
   even if no Term is matched). Also good for broad recall on procedure / how-does
   intents.
   args: {{"query": "<paraphrased sub-question>", "top_k": 8}}

2. cypher_query — write a single read-only Cypher to fill a SPECIFIC gap. Best
   when you know a Term/spec_id and want section-precise lookup (definitions,
   abbreviations, named procedures).
   args: {{"cypher": "<MATCH ... RETURN chunk_id, content, spec_id, section, score LIMIT $top_k>",
            "top_k": 6, "purpose": "<which gap this targets>"}}
   Constraints: read-only, single statement, MUST return columns
                chunk_id, content, spec_id, section, score, MUST use $top_k for LIMIT.

3. expand_term — fetch full_name + source_specs for a term not yet resolved.
   args: {{"abbreviation": "<UPPER>"}}

4. inspect_chunk — read full content of one promising chunk to verify relevance.
   args: {{"chunk_id": "<id from chunks above>"}}

5. finish — stop and return what you have.
   args: {{"reason": "<short why-enough sentence>"}}

# Strategy hints
- ITER 1 with NO prior chunks: usually start with `vector_search` (semantic) or
  `cypher_query` if the gap maps cleanly to a Term/section_title pattern. Avoid
  `inspect_chunk` (no chunks to inspect) and `finish` (you have nothing yet).
- LATER ITERS: read the chunks list above. If a chunk already covers the gap,
  call `finish`. If gaps remain and Cypher with the gold pattern keeps returning
  abbreviations, switch to `vector_search` with a paraphrased sub-question, or
  write a Cypher with `section_title CONTAINS '<concept>'` instead of source_specs.
- Don't repeat the same Cypher twice. Vary the angle each iter.

# Output format — STRICT JSON, no prose, no fences, no comments
{{"thought": "<one sentence on why this action>",
  "tool": "<one of: vector_search | cypher_query | expand_term | inspect_chunk | finish>",
  "args": {{...}},
  "remaining_gaps": ["<gap still open after this action, if any>", ...]}}

Pick `finish` when the chunks above already cover all gaps — quality > diversity.
"""


# Gap-seeding is now handled inside the research-planning loop (the LLM emits
# its final gaps via the `finish` tool action). The static gap-seeding prompt
# has been removed in favour of `build_research_planner_prompt` below.


# Tool names allowed during the research-planning phase (a small ReAct loop the
# LLM uses to inspect the KG before producing the final gap list).
RESEARCH_TOOLS = ("kg_search_titles", "kg_search_terms", "kg_search_specs", "finish")


# Format the cumulative tool-call history (research planning) into a compact
# block for the next iter's prompt — each entry shows tool, args, and a short
# observation summary so the LLM can avoid repeating queries.
def _research_history_block(history: list[dict]) -> str:
    if not history:
        return "(no KG queries yet — first iteration)"
    lines = []
    for h in history:
        tool = h.get("tool", "?")
        args_brief = _truncate(json.dumps(h.get("args") or {}, ensure_ascii=False), 80)
        obs_lines = h.get("observation_lines") or []
        if obs_lines:
            lines.append(f"  iter {h.get('iter')}: {tool}({args_brief})")
            for ol in obs_lines[:8]:
                lines.append(f"      • {_truncate(ol, 110)}")
            if len(obs_lines) > 8:
                lines.append(f"      … and {len(obs_lines) - 8} more")
        else:
            obs = _truncate(str(h.get("observation_summary") or ""), 100)
            lines.append(f"  iter {h.get('iter')}: {tool}({args_brief}) → {obs}")
    return "\n".join(lines)


# Research-planning prompt — the LLM picks ONE KG-inspection action per iter,
# then emits `finish` with the final gap list. Replaces the static
# canonicalization+gap-seed pipeline; no hardcoded term examples.
def build_research_planner_prompt(
    *,
    question: str,
    intent: str,
    resolved_terms: Optional[dict],
    history: list[dict],
    iter_idx: int,
    max_iter: int,
    last_error: Optional[str] = None,
) -> str:
    error_block = ""
    if last_error:
        error_block = (
            f"\n# Previous action failed\n"
            f"Reason: {last_error}\n"
            f"Pick a different action — do not repeat the same call.\n"
        )

    return f"""You are a 3GPP retrieval planner. Goal: produce the FINAL list of sub-questions
("research gaps") that, when answered with chunks from the knowledge graph, will fully cover
the user's question.

# User question
{question}

# Intent
{intent}

# Resolved terms (authoritative from KG)
{_resolved_terms_block(resolved_terms)}

# Why you have tools
The user may use INFORMAL wording for concepts that have a different spec term
(e.g. "traffic steering" vs "Traffic Influence", "QoS routing" vs "URSP").
You can inspect the KG to verify the canonical phrasing BEFORE you commit to a gap list.
You also might NOT need to search at all — if the question already uses standard
terminology you've worked with, call `finish` immediately.

# Live KG content you can query
- 197k Chunks  (section_title, content, spec_id, chunk_type)
- 18k Terms    (abbreviation, full_name, source_specs, primary_spec)
- 1.6k Documents (spec_id)

# Available tools (pick exactly ONE per turn)
1. kg_search_titles — search Chunk.section_title for a substring (case-insensitive),
   returns up to 20 distinct section_titles with their spec_ids. Use to confirm or
   discover canonical spec phrasing.
   args: {{"keyword": "<phrase or word>", "limit": 20}}

2. kg_search_terms — search Term nodes by abbreviation OR full_name (case-insensitive
   substring), returns up to 20 (abbreviation, full_name, primary_spec) tuples.
   Use to find a term you suspect exists under a different name.
   args: {{"keyword": "<phrase or word>", "limit": 20}}

3. kg_search_specs — search Document.spec_id for a substring, returns up to 10
   spec_ids. Use to verify a spec reference like "TS 23.502" exists.
   args: {{"keyword": "<phrase>", "limit": 10}}

4. finish — emit the final gap list and stop the research-planning phase.
   args: {{"gaps": ["<sub-question 1>", "<sub-question 2>", ...]}}

# Strategy
- Compare-style questions ("compare A and B in aspects X, Y, Z"): produce ONE gap per
  (entity × aspect). Do NOT drop aspects to fit any count.
- Use canonical spec terms in the gaps once you've verified them via search.
- Don't over-search. 0-3 tool calls is typical, then finish.
- Each gap must be answerable by ONE focused chunk lookup.

# Current state
- Iteration: {iter_idx} of {max_iter}
- KG queries made so far:
{_research_history_block(history)}
{error_block}
# Output format — STRICT JSON, no prose, no fences, no comments
{{"thought": "<one sentence on why this action>",
  "tool": "<one of: kg_search_titles | kg_search_terms | kg_search_specs | finish>",
  "args": {{...}}}}
"""


# Parse a research-planning action emitted by the LLM. Reuses the JSON object
# extractor + applies a different allowed-tool list than the main planner.
def parse_research_action(raw: str) -> PlannerAction:
    text = _extract_json_object(raw)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise PlannerParseError(f"Invalid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise PlannerParseError("Planner output is not a JSON object")

    tool = str(obj.get("tool", "")).strip()
    if tool not in RESEARCH_TOOLS:
        raise PlannerParseError(f"Unknown research tool {tool!r}; must be one of {RESEARCH_TOOLS}")

    args = obj.get("args") or {}
    if not isinstance(args, dict):
        raise PlannerParseError("`args` must be a JSON object")

    return PlannerAction(
        thought=str(obj.get("thought") or "").strip(),
        tool=tool,
        args=args,
        remaining_gaps=[],  # research planner doesn't use this field
        raw=raw,
    )


# Strip code fences and pull the first balanced {...} from the LLM output.
# Also tolerates LLMs that drop the trailing `}` / `]` after closing the last
# array (qwen3:14b sometimes does this) — _balance_json appends matching
# closers when the brace/bracket count is uneven.
def _extract_json_object(raw: str) -> str:
    text = _FENCE.sub("", raw or "").strip()
    if text.startswith("{"):
        return _balance_json(text)
    m = _FIRST_OBJ.search(text)
    if not m:
        raise PlannerParseError(f"No JSON object found in LLM output: {raw[:200]!r}")
    return _balance_json(m.group(0))


# Walk the JSON text once, tracking string state, and append any missing
# closing `}` / `]` in the right order so json.loads can succeed. Cheap
# recovery for truncated LLM streams; if the input is already balanced it
# returns unchanged.
def _balance_json(text: str) -> str:
    stack: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()
    if not stack:
        return text
    closers = "".join("}" if c == "{" else "]" for c in reversed(stack))
    return text + closers


# Strip fences and pull the first JSON array from the LLM output.
def _extract_json_array(raw: str) -> str:
    text = _FENCE.sub("", raw or "").strip()
    if text.startswith("["):
        return text
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        raise PlannerParseError(f"No JSON array found in LLM output: {raw[:200]!r}")
    return m.group(0)


# Validate + normalise a parsed planner action.
def parse_planner_action(raw: str) -> PlannerAction:
    text = _extract_json_object(raw)
    try:
        obj: Any = json.loads(text)
    except json.JSONDecodeError as e:
        raise PlannerParseError(f"Invalid JSON: {e}") from e

    if not isinstance(obj, dict):
        raise PlannerParseError("Planner output is not a JSON object")

    tool = str(obj.get("tool", "")).strip()
    if tool not in ALLOWED_TOOLS:
        raise PlannerParseError(f"Unknown tool {tool!r}; must be one of {ALLOWED_TOOLS}")

    args = obj.get("args") or {}
    if not isinstance(args, dict):
        raise PlannerParseError("`args` must be a JSON object")

    gaps = obj.get("remaining_gaps") or []
    if not isinstance(gaps, list):
        gaps = []
    gaps = [str(g) for g in gaps if isinstance(g, (str, int, float))]

    return PlannerAction(
        thought=str(obj.get("thought") or "").strip(),
        tool=tool,
        args=args,
        remaining_gaps=gaps,
        raw=raw,
    )


# Parse the gap-seeding step's JSON array of sub-questions.
def parse_gap_list(raw: str) -> list[str]:
    text = _extract_json_array(raw)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise PlannerParseError(f"Invalid JSON array: {e}") from e
    if not isinstance(obj, list):
        raise PlannerParseError("Expected JSON array of strings")
    return [str(x).strip() for x in obj if isinstance(x, (str, int, float)) and str(x).strip()]
