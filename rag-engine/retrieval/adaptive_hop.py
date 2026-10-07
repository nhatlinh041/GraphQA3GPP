"""
Adaptive ReAct retrieval agent.

The LLM acts as a retrieval planner that decides ONE next action per iteration:
generate Cypher, expand a term, fall back to vector search, inspect a chunk,
or stop. The orchestrator yields SSE events so the UI can render each step
in the ThinkingTrail.

Hard budget: max iterations + wall-clock latency. The LLM is encouraged to
finish early; the system enforces a stop when budgets are exceeded.
"""
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from neo4j import GraphDatabase

from llm import OllamaClient
from retrieval.adaptive_hop_prompts import (
    PlannerAction,
    PlannerParseError,
    build_planner_prompt,
    build_research_planner_prompt,
    parse_planner_action,
    parse_research_action,
)
from retrieval.cypher_generator import CypherValidationError, LLMCypherGenerator
from retrieval.vector_search import VectorSearcher

if TYPE_CHECKING:
    # Avoid circular import: adaptive_hop is imported via retrieval/__init__,
    # which is imported by pipeline/orchestrator.py — pipeline.term_index is
    # safe to import lazily but TYPE_CHECKING keeps the static type ref.
    from pipeline.term_index import TermIndex


# Max chars to keep per chunk content (avoid blowing the planner prompt).
CHUNK_CONTENT_CAP = 2000

# Cap iterations of the research-planning sub-loop. Each iter is one LLM call
# that picks ONE KG-inspection tool (or `finish`). The LLM is encouraged to
# call finish early.
MAX_RESEARCH_ITER = 5


# Normalise a chunk_id so the duplicate spec_id formats produced during KG
# building (`ts_23.288_4.1` vs `ts_23_288_4.1` — same content, two nodes) are
# treated as the SAME logical chunk during dedup.
def normalize_chunk_id(cid: Optional[str]) -> Optional[str]:
    if not cid:
        return None
    return cid.replace(".", "_")


@dataclass
class HopState:
    question: str
    intent: str
    seeds: list[str]
    resolved_terms: dict
    # Chunks the adaptive loop NEWLY collected (returned for fusion).
    chunks: list[dict] = field(default_factory=list)
    # Chunks passed in from prior stages (vector+graph). Shown to the planner so it
    # knows what's already covered, but NOT returned for fusion (would double-count in RRF).
    prior_chunks: list[dict] = field(default_factory=list)
    seen_ids: set[str] = field(default_factory=set)
    gaps: list[str] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    iter: int = 0
    elapsed_ms: int = 0
    last_error: Optional[str] = None
    finish_reason: Optional[str] = None

    # Seed prior chunks: fill seen_ids so we dedup against them, but keep them
    # in a separate list so the loop's own contribution is what we return.
    # Dedup uses NORMALISED chunk_id so the `ts_23.288_4.1` / `ts_23_288_4.1`
    # duplicates produced during KG ingestion collapse into one slot.
    def seed_prior(self, prior: list[dict]) -> None:
        for c in prior:
            norm = normalize_chunk_id(c.get("chunk_id"))
            if not norm or norm in self.seen_ids:
                continue
            content = c.get("content") or ""
            if len(content) > CHUNK_CONTENT_CAP:
                c = {**c, "content": content[:CHUNK_CONTENT_CAP]}
            self.prior_chunks.append(c)
            self.seen_ids.add(norm)

    # Add new chunks while skipping duplicates (by normalised chunk_id).
    # Returns the count actually added.
    def add_chunks(self, new_chunks: list[dict]) -> int:
        added = 0
        for c in new_chunks:
            norm = normalize_chunk_id(c.get("chunk_id"))
            if not norm or norm in self.seen_ids:
                continue
            content = c.get("content") or ""
            if len(content) > CHUNK_CONTENT_CAP:
                c = {**c, "content": content[:CHUNK_CONTENT_CAP]}
            self.chunks.append(c)
            self.seen_ids.add(norm)
            added += 1
        return added

    # Combined view (prior + newly collected) for the planner prompt.
    def all_chunks_view(self) -> list[dict]:
        return self.prior_chunks + self.chunks


class AdaptiveHopSearcher:
    """LLM-driven adaptive retrieval. Reuses Cypher sanitizer + vector + term resolver."""

    def __init__(
        self,
        *,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        llm: OllamaClient,
        planner_model: str,
        vector_searcher: VectorSearcher,
        cypher_generator: LLMCypherGenerator,
        term_index: Optional["TermIndex"] = None,
        max_iter: int = 4,
        budget_ms: int = 8000,
        max_chunks: int = 30,
    ):
        self._driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self._llm = llm
        self._planner_model = planner_model
        self._vector = vector_searcher
        self._cypher_gen = cypher_generator
        # In-memory Term snapshot. Replaces per-call Neo4j round-trips for
        # `expand_term` and `kg_search_terms` tools. Optional for backward
        # compat with tests that don't need term lookups.
        self._term_index = term_index
        self._max_iter = max_iter
        self._budget_ms = budget_ms
        self._max_chunks = max_chunks

    def close(self) -> None:
        self._driver.close()

    # ------------------------------------------------------------------
    # Public streaming API — yields SSE-style event dicts for the UI.
    # ------------------------------------------------------------------

    def search_streaming(
        self,
        *,
        question: str,
        intent: str,
        seeds: list[str],
        resolved_terms: Optional[dict] = None,
        prior_chunks: Optional[list[dict]] = None,
        think: bool = True,
        model: Optional[str] = None,
        num_ctx: Optional[int] = None,
    ) -> Iterator[dict]:
        """
        Run the ReAct loop and yield events. The final `hop_finish` event carries
        the full chunk list under `data.chunks` so the orchestrator can pull it
        for fusion.
        """
        state = HopState(
            question=question,
            intent=intent,
            seeds=seeds,
            resolved_terms=resolved_terms or {},
        )
        if prior_chunks:
            state.seed_prior(prior_chunks)

        # Single-model run: planner + Cypher gen use the request's model when
        # provided, else fall back to the construction-time default.
        active_model = model or self._planner_model
        t0 = time.monotonic()

        # --- Phase 0: research planning (KG-aware, tool-using) --------------
        # The planner LLM picks 0–N KG inspection tools (kg_search_titles /
        # kg_search_terms / kg_search_specs) to verify canonical spec terminology
        # before emitting the final gap list via `finish`. Replaces the old
        # static canonicalize+gap-seed prompts (no hardcoded examples).
        yield {
            "stage": "hop_research_start",
            "data": {
                "input": {
                    "question": question,
                    "intent": intent,
                    "seeds": seeds,
                    "max_iter": self._max_iter,
                    "max_research_iter": MAX_RESEARCH_ITER,
                    "resolved_terms": resolved_terms or {},
                },
            },
        }

        research_history: list[dict] = []
        research_last_error: Optional[str] = None
        finish_emitted = False
        for r_iter in range(1, MAX_RESEARCH_ITER + 1):
            r_prompt = build_research_planner_prompt(
                question=question,
                intent=intent,
                resolved_terms=resolved_terms or {},
                history=research_history,
                iter_idx=r_iter,
                max_iter=MAX_RESEARCH_ITER,
                last_error=research_last_error,
            )
            research_last_error = None

            yield {
                "stage": "hop_research_iter_start",
                "data": {
                    "iter": r_iter,
                    "input": {"prompt": r_prompt},
                },
            }

            r_raw_parts: list[str] = []
            r_thinking_parts: list[str] = []
            try:
                for ev in self._llm.generate_stream_full(
                    r_prompt, model=active_model, think=think, num_ctx=num_ctx
                ):
                    if ev["kind"] == "thinking":
                        r_thinking_parts.append(ev["token"])
                        yield {
                            "stage": "hop_research_thinking",
                            "data": {
                                "iter": r_iter,
                                "token": ev["token"],
                                "accumulated": "".join(r_thinking_parts),
                            },
                        }
                    else:
                        r_raw_parts.append(ev["token"])
                        yield {
                            "stage": "hop_research_token",
                            "data": {
                                "iter": r_iter,
                                "token": ev["token"],
                                "accumulated": "".join(r_raw_parts),
                            },
                        }
                r_raw = "".join(r_raw_parts)
                action = parse_research_action(r_raw)
            except PlannerParseError as e:
                research_last_error = f"parse_error: {e}"
                yield {"stage": "hop_warning", "data": {
                    "phase": "research_parse",
                    "iter": r_iter,
                    "error": str(e),
                    "raw": ("".join(r_raw_parts))[:500],
                }}
                research_history.append({
                    "iter": r_iter,
                    "tool": "(parse_error)",
                    "args": {},
                    "observation_summary": str(e)[:120],
                })
                if (
                    len(research_history) >= 2
                    and research_history[-2].get("tool") == "(parse_error)"
                ):
                    break
                continue
            except Exception as e:
                yield {"stage": "hop_warning", "data": {
                    "phase": "research_llm",
                    "iter": r_iter,
                    "error": f"{type(e).__name__}: {e}",
                }}
                break

            yield {
                "stage": "hop_research_decision",
                "data": {
                    "iter": r_iter,
                    "thought": action.thought,
                    "tool": action.tool,
                    "args": action.args,
                },
            }

            if action.tool == "finish":
                gaps_arg = action.args.get("gaps") or []
                if isinstance(gaps_arg, list):
                    cleaned = [str(g).strip() for g in gaps_arg if str(g).strip()]
                    state.gaps = cleaned[:8] if cleaned else [question]
                else:
                    state.gaps = [question]
                research_history.append({
                    "iter": r_iter,
                    "tool": "finish",
                    "args": action.args,
                    "observation_summary": f"emitted {len(state.gaps)} gaps",
                })
                yield {
                    "stage": "hop_research_act",
                    "data": {
                        "iter": r_iter,
                        "tool": "finish",
                        "input": action.args,
                        "output": {"gaps": state.gaps, "count": len(state.gaps)},
                    },
                }
                finish_emitted = True
                break

            # Execute KG-search tool
            obs_lines, summary = self._exec_kg_tool(action.tool, action.args)
            yield {
                "stage": "hop_research_act",
                "data": {
                    "iter": r_iter,
                    "tool": action.tool,
                    "input": action.args,
                    "output": {"lines": obs_lines, "summary": summary},
                    "count": len(obs_lines),
                },
            }
            research_history.append({
                "iter": r_iter,
                "tool": action.tool,
                "args": action.args,
                "observation_lines": obs_lines,
                "observation_summary": summary,
            })

        # Fallback: if the LLM never emitted finish, use the question itself.
        if not state.gaps:
            state.gaps = [question]
            yield {"stage": "hop_warning", "data": {
                "phase": "research_no_finish",
                "info": "research-planning ended without a finish action; falling back to single-gap = original question",
            }}

        # Bookend event the rest of the pipeline (and the UI) treats as the
        # "research planning complete" signal — carries the final gap list.
        yield {
            "stage": "hop_research_done",
            "data": {
                "input": {
                    "question": question,
                    "intent": intent,
                    "iters": research_history,
                },
                "output": {
                    "gaps": state.gaps,
                    "iters": len(research_history),
                    "finish_emitted": finish_emitted,
                    "prior_chunks": len(state.prior_chunks),
                },
                "gaps": state.gaps,
                "count": len(state.gaps),
            },
        }

        # --- Step 1..N: ReAct iterations -----------------------------------
        while True:
            state.iter += 1
            state.elapsed_ms = int((time.monotonic() - t0) * 1000)

            # Hard budget: iter cap.
            if state.iter > self._max_iter:
                state.finish_reason = "max_iter_reached"
                break

            # Hard budget: latency cap. Only enforced when budget_ms > 0;
            # set to 0 (or negative) to disable wall-clock cap entirely.
            if self._budget_ms > 0 and state.elapsed_ms >= self._budget_ms:
                state.finish_reason = "budget_ms_exceeded"
                break

            # Hard budget: chunk cap (counts loop-added chunks only — prior chunks
            # were already retrieved upstream and don't count against this cap).
            if len(state.chunks) >= self._max_chunks:
                state.finish_reason = "max_chunks_reached"
                break

            # Announce planning step (UI shows "Iter N: planning…").
            yield {
                "stage": "hop_plan",
                "data": {
                    "iter": state.iter,
                    "elapsed_ms": state.elapsed_ms,
                    "chunks_so_far": len(state.chunks),
                    "prior_chunks": len(state.prior_chunks),
                    "gaps": state.gaps,
                    "history_len": len(state.history),
                },
            }

            # Build planner prompt + stream the action. Show planner the full view
            # (prior + new) so it knows what's already covered.
            prompt = build_planner_prompt(
                question=question,
                intent=intent,
                resolved_terms=resolved_terms or {},
                seeds=seeds,
                chunks=state.all_chunks_view(),
                gaps=state.gaps,
                history=state.history,
                iter_idx=state.iter,
                max_iter=self._max_iter,
                last_error=state.last_error,
            )
            state.last_error = None

            action: Optional[PlannerAction] = None
            raw_parts: list[str] = []
            thinking_parts: list[str] = []
            try:
                for ev in self._llm.generate_stream_full(
                    prompt, model=active_model, think=think, num_ctx=num_ctx
                ):
                    if ev["kind"] == "thinking":
                        thinking_parts.append(ev["token"])
                        yield {
                            "stage": "hop_thinking",
                            "data": {
                                "iter": state.iter,
                                "token": ev["token"],
                            },
                        }
                    else:
                        raw_parts.append(ev["token"])
                        yield {
                            "stage": "hop_planner_token",
                            "data": {
                                "iter": state.iter,
                                "token": ev["token"],
                            },
                        }
                raw = "".join(raw_parts)
                action = parse_planner_action(raw)
            except PlannerParseError as e:
                # Re-prompt strategy: don't loop forever on bad JSON. Mark error,
                # let the next iter see it, but if it happens twice in a row → finish.
                state.last_error = f"planner_parse_error: {e}"
                yield {
                    "stage": "hop_warning",
                    "data": {
                        "iter": state.iter,
                        "phase": "parse_action",
                        "error": str(e),
                        "raw": ("".join(raw_parts))[:500],
                    },
                }
                state.history.append({
                    "iter": state.iter,
                    "tool": "(parse_error)",
                    "args": {},
                    "observation_summary": str(e)[:120],
                })
                # If two parse errors in a row → cut losses.
                if (
                    len(state.history) >= 2
                    and state.history[-2].get("tool") == "(parse_error)"
                ):
                    state.finish_reason = "consecutive_parse_errors"
                    break
                continue
            except Exception as e:
                # LLM call itself blew up — surface and stop.
                state.finish_reason = f"planner_llm_error: {type(e).__name__}: {e}"
                break

            yield {
                "stage": "hop_decision",
                "data": {
                    "iter": state.iter,
                    "thought": action.thought,
                    "tool": action.tool,
                    "args": action.args,
                    "remaining_gaps": action.remaining_gaps,
                },
            }

            # --- Execute the chosen action --------------------------------
            if action.tool == "finish":
                state.finish_reason = action.args.get("reason") or "llm_finish"
                state.history.append({
                    "iter": state.iter,
                    "tool": "finish",
                    "args": action.args,
                    "observation_summary": "stop",
                })
                break

            tool_event, observation_summary = self._execute_tool(action, state)
            yield tool_event

            state.history.append({
                "iter": state.iter,
                "tool": action.tool,
                "args": action.args,
                "observation_summary": observation_summary,
            })

            # Update gaps if the planner emitted a new view.
            if action.remaining_gaps:
                state.gaps = action.remaining_gaps

        # --- Final summary event -------------------------------------------
        state.elapsed_ms = int((time.monotonic() - t0) * 1000)
        yield {
            "stage": "hop_finish",
            "data": {
                "iters": state.iter - (1 if state.iter > self._max_iter else 0),
                "elapsed_ms": state.elapsed_ms,
                "reason": state.finish_reason or "unknown",
                "total_chunks": len(state.chunks),
                "gaps_remaining": state.gaps,
                "chunks": state.chunks,
            },
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    # Execute a research-planning KG inspection tool. Returns (lines, summary).
    # Read-only by construction; queries are parameterised, no string interpolation.
    def _exec_kg_tool(self, tool: str, args: dict) -> tuple[list[str], str]:
        keyword = (args.get("keyword") or "").strip()
        limit = int(args.get("limit") or 20)
        limit = max(1, min(limit, 50))
        if not keyword:
            return ([], "error: missing 'keyword' arg")

        try:
            with self._driver.session(default_access_mode="READ") as session:
                if tool == "kg_search_titles":
                    cypher = (
                        "MATCH (c:Chunk) "
                        "WHERE toLower(c.section_title) CONTAINS toLower($kw) "
                        "RETURN DISTINCT c.section_title AS title, c.spec_id AS spec "
                        "ORDER BY spec, title LIMIT $lim"
                    )
                    rows = session.run(cypher, kw=keyword, lim=limit).data()
                    lines = [f"[{r['spec']}] {r['title']}" for r in rows if r.get("title")]
                    return (lines, f"{len(lines)} titles for {keyword!r}")

                if tool == "kg_search_terms":
                    # In-memory substring search (replaces live Cypher). Uses
                    # the same index loaded once at startup, so this is O(N)
                    # over ~31k records but no network round-trip.
                    if self._term_index is None:
                        return ([], "term_index_unavailable")
                    rows = self._term_index.search(keyword, limit=limit)
                    lines = [
                        f"{r['abbreviation']} = {r['full_name']}"
                        + (f" (primary: {r['primary_spec']})" if r.get("primary_spec") else "")
                        for r in rows
                        if r.get("abbreviation") or r.get("full_name")
                    ]
                    return (lines, f"{len(lines)} terms for {keyword!r}")

                if tool == "kg_search_specs":
                    cypher = (
                        "MATCH (d:Document) "
                        "WHERE toLower(d.spec_id) CONTAINS toLower($kw) "
                        "RETURN DISTINCT d.spec_id AS spec ORDER BY spec LIMIT $lim"
                    )
                    rows = session.run(cypher, kw=keyword, lim=limit).data()
                    lines = [r["spec"] for r in rows if r.get("spec")]
                    return (lines, f"{len(lines)} specs for {keyword!r}")
        except Exception as e:
            return ([], f"error: {type(e).__name__}: {e}")

        return ([], f"error: unknown tool {tool!r}")

    # Dispatch the planner-chosen tool, return (event_dict, short_observation_summary).
    def _execute_tool(
        self, action: PlannerAction, state: HopState
    ) -> tuple[dict, str]:
        tool = action.tool
        args = action.args
        try:
            if tool == "cypher_query":
                return self._tool_cypher(args, state)
            if tool == "expand_term":
                return self._tool_expand_term(args, state)
            if tool == "vector_search":
                return self._tool_vector(args, state)
            if tool == "inspect_chunk":
                return self._tool_inspect(args, state)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            state.last_error = f"{tool}_error: {err}"
            return (
                {
                    "stage": "hop_act",
                    "data": {
                        "iter": state.iter,
                        "tool": tool,
                        "input": args,
                        "output": {"error": err},
                        "added": 0,
                    },
                },
                f"error: {err[:100]}",
            )

        # Unreachable — parser already restricts to ALLOWED_TOOLS.
        return (
            {
                "stage": "hop_act",
                "data": {
                    "iter": state.iter,
                    "tool": tool,
                    "input": args,
                    "output": {"error": "unknown_tool"},
                    "added": 0,
                },
            },
            "error: unknown_tool",
        )

    # cypher_query — validate + run the LLM-written Cypher.
    def _tool_cypher(self, args: dict, state: HopState) -> tuple[dict, str]:
        cypher = (args.get("cypher") or "").strip()
        top_k = int(args.get("top_k") or 6)
        purpose = args.get("purpose") or ""

        if not cypher:
            raise ValueError("missing 'cypher' arg")

        # Reuse the existing sanitizer — same rules as graph search.
        try:
            self._cypher_gen._validate(cypher)  # noqa: SLF001 — same package
        except CypherValidationError as e:
            state.last_error = f"cypher_rejected: {e}"
            return (
                {
                    "stage": "hop_act",
                    "data": {
                        "iter": state.iter,
                        "tool": "cypher_query",
                        "input": {"cypher": cypher, "top_k": top_k, "purpose": purpose},
                        "output": {"error": f"validation: {e}"},
                        "added": 0,
                    },
                },
                f"cypher_rejected: {str(e)[:100]}",
            )

        rows: list[dict] = []
        # Outer budget_ms is the real timeout safeguard; we don't set a per-query
        # transaction timeout to keep the API simple and version-agnostic.
        with self._driver.session(default_access_mode="READ") as session:
            try:
                result = session.run(cypher, top_k=top_k)
                for record in result:
                    rows.append(self._normalise_row(dict(record)))
            except Exception as e:
                state.last_error = f"cypher_run_error: {e}"
                return (
                    {
                        "stage": "hop_act",
                        "data": {
                            "iter": state.iter,
                            "tool": "cypher_query",
                            "input": {"cypher": cypher, "top_k": top_k, "purpose": purpose},
                            "output": {"error": f"run: {e}"},
                            "added": 0,
                        },
                    },
                    f"cypher_run_error: {str(e)[:100]}",
                )

        added = state.add_chunks(rows)
        preview = _preview_chunks(rows, n=3)
        return (
            {
                "stage": "hop_act",
                "data": {
                    "iter": state.iter,
                    "tool": "cypher_query",
                    "input": {"cypher": cypher, "top_k": top_k, "purpose": purpose},
                    "output": {
                        "rows": len(rows),
                        "added": added,
                        "top": preview,
                    },
                    "added": added,
                    # Top-level `top` mirrors the convention used by the other
                    # retrieval stages so the UI's ChunkPreviewList finds it.
                    "top": preview,
                    "count": len(rows),
                },
            },
            f"rows={len(rows)} added={added}",
        )

    # expand_term — fetch full_name + source_specs for a term.
    def _tool_expand_term(self, args: dict, state: HopState) -> tuple[dict, str]:
        abbr = (args.get("abbreviation") or "").strip().upper()
        if not abbr:
            raise ValueError("missing 'abbreviation' arg")

        # In-memory lookup (replaces live Cypher). Index is loaded once at
        # startup, so this is O(1).
        rec = self._term_index.lookup_abbrev(abbr) if self._term_index else None

        if not rec:
            return (
                {
                    "stage": "hop_act",
                    "data": {
                        "iter": state.iter,
                        "tool": "expand_term",
                        "input": args,
                        "output": {"found": False},
                        "added": 0,
                    },
                },
                f"term_not_found: {abbr}",
            )

        info = {
            "abbreviation": rec["abbreviation"],
            "full_name": rec["full_name"],
            "source_specs": list(rec.get("source_specs") or []),
            "primary_spec": rec.get("primary_spec"),
        }
        # Merge into resolved_terms so future planner prompts see it.
        state.resolved_terms[info["abbreviation"]] = {
            "full_name": info["full_name"],
            "specs": info["source_specs"],
        }
        return (
            {
                "stage": "hop_act",
                "data": {
                    "iter": state.iter,
                    "tool": "expand_term",
                    "input": args,
                    "output": {"found": True, **info},
                    "added": 0,
                },
            },
            f"resolved {abbr} → {info['full_name']}",
        )

    # vector_search — fallback semantic search when graph is dry.
    def _tool_vector(self, args: dict, state: HopState) -> tuple[dict, str]:
        query = (args.get("query") or "").strip()
        top_k = int(args.get("top_k") or 6)
        if not query:
            raise ValueError("missing 'query' arg")

        rows = self._vector.search(query, top_k=top_k)
        added = state.add_chunks(rows)
        preview = _preview_chunks(rows, n=3)
        return (
            {
                "stage": "hop_act",
                "data": {
                    "iter": state.iter,
                    "tool": "vector_search",
                    "input": {"query": query, "top_k": top_k},
                    "output": {
                        "rows": len(rows),
                        "added": added,
                        "top": preview,
                    },
                    "added": added,
                    # Top-level `top` so the UI ChunkPreviewList renders inline.
                    "top": preview,
                    "count": len(rows),
                },
            },
            f"vector rows={len(rows)} added={added}",
        )

    # inspect_chunk — return full content of a single chunk by id.
    def _tool_inspect(self, args: dict, state: HopState) -> tuple[dict, str]:
        chunk_id = (args.get("chunk_id") or "").strip()
        if not chunk_id:
            raise ValueError("missing 'chunk_id' arg")

        cypher = (
            "MATCH (c:Chunk {chunk_id: $cid}) "
            "RETURN c.chunk_id AS chunk_id, c.content AS content, c.spec_id AS spec_id, "
            "       c.section_title AS section, c.chunk_type AS chunk_type"
        )
        with self._driver.session() as session:
            rec = session.run(cypher, cid=chunk_id).single()

        if not rec:
            return (
                {
                    "stage": "hop_act",
                    "data": {
                        "iter": state.iter,
                        "tool": "inspect_chunk",
                        "input": args,
                        "output": {"found": False},
                        "added": 0,
                    },
                },
                f"chunk_not_found: {chunk_id}",
            )

        info = self._normalise_row(dict(rec))
        # Already-known chunk → just surface; don't duplicate-add.
        if chunk_id in state.seen_ids:
            return (
                {
                    "stage": "hop_act",
                    "data": {
                        "iter": state.iter,
                        "tool": "inspect_chunk",
                        "input": args,
                        "output": {"found": True, "already_known": True, **info},
                        "added": 0,
                    },
                },
                f"inspect known {chunk_id}",
            )

        added = state.add_chunks([info])
        return (
            {
                "stage": "hop_act",
                "data": {
                    "iter": state.iter,
                    "tool": "inspect_chunk",
                    "input": args,
                    "output": {"found": True, "already_known": False, **info},
                    "added": added,
                },
            },
            f"inspected {chunk_id}",
        )

    # Map planner-Cypher output columns to the canonical chunk dict shape.
    @staticmethod
    def _normalise_row(row: dict) -> dict:
        # Section may come back under either alias.
        section = row.get("section") or row.get("section_title")
        return {
            "chunk_id": row.get("chunk_id"),
            "content": row.get("content") or "",
            "spec_id": row.get("spec_id"),
            "section": section,
            "chunk_type": row.get("chunk_type"),
            "score": row.get("score"),
        }


# Compact preview of chunks for the SSE event payload — UI-friendly, not for prompts.
def _preview_chunks(chunks: list[dict], n: int = 3, preview_chars: int = 100) -> list[dict]:
    out = []
    for c in chunks[:n]:
        content = (c.get("content") or "").strip().replace("\n", " ")
        if len(content) > preview_chars:
            content = content[:preview_chars].rstrip() + "…"
        out.append({
            "spec_id": c.get("spec_id", "?"),
            "section": c.get("section") or c.get("section_title") or "?",
            "score": c.get("score"),
            "preview": content,
        })
    return out
