"""
Graph search via LLM-generated Cypher.
Templates are gone — the LLM sees the live KG schema and writes a Cypher query
that targets exactly what's in the graph.

`search_streaming` yields progress events so the UI can show the LLM writing
the Cypher token-by-token (claude.ai-style live tool call).
"""
import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Optional

from neo4j import GraphDatabase

from retrieval.cypher_generator import LLMCypherGenerator
from retrieval.schema_introspect import SchemaIntrospector


# Defensive: the LLM may forget to alias columns, so a query like
#   RETURN chunk.chunk_id, chunk.content, ...
# produces dict keys like "chunk.chunk_id" instead of "chunk_id". Look up each
# logical field by trying its expected alias first, then any key that ends in
# ".<field>" (handles `c.chunk_id`, `chunk.chunk_id`, etc.) before giving up.
def _pick(raw: dict, *names: str):
    for n in names:
        if n in raw and raw[n] is not None:
            return raw[n]
    # Fallback: any key whose last dotted segment matches one of the names
    name_set = set(names)
    for k, v in raw.items():
        if v is None:
            continue
        tail = k.rsplit(".", 1)[-1]
        if tail in name_set:
            return v
    return None


def _normalize_chunk(raw: dict, fallback_idx: int) -> dict:
    chunk_id = _pick(raw, "chunk_id")
    content = _pick(raw, "content", "text")
    spec_id = _pick(raw, "spec_id", "specId", "specification_id")
    section = _pick(raw, "section", "section_title", "section_id")
    score = _pick(raw, "score")

    if chunk_id in (None, ""):
        # Build a stable surrogate id so RRF doesn't merge unrelated chunks under None
        seed = f"{spec_id or '?'}::{section or '?'}::{fallback_idx}::{(content or '')[:60]}"
        chunk_id = "graph_" + hashlib.md5(seed.encode()).hexdigest()[:12]

    if not isinstance(content, str):
        content = str(content) if content is not None else ""

    out = {
        "chunk_id": str(chunk_id),
        "content": content,
        # Coerce to str: an LLM-written query may RETURN a number in these
        # columns (gemma3 returned `section` = 1.0 on KGF4-1119, v59), and the
        # rerank path calls .lower()/.strip() on them.
        "spec_id": str(spec_id) if spec_id not in (None, "") else "?",
        "section": str(section) if section not in (None, "") else "?",
        "score": float(score) if isinstance(score, (int, float)) else 0.0,
    }
    # Optional structural signals some patterns emit (FG: chunk_type +
    # has_step_all_actors / op_prefix_match) — feed the Part E rerank bias
    # (rerank_bias.compute_bias). Pass through only when present.
    for opt in ("chunk_type", "has_step_all_actors", "op_prefix_match"):
        v = _pick(raw, opt)
        if v is not None:
            out[opt] = v
    return out


@dataclass
class GraphSearchResult:
    """Final result of graph search — surfaced in the SSE event."""
    chunks: list[dict] = field(default_factory=list)
    cypher: str = ""
    cypher_raw: str = ""
    prompt: str = ""
    model: str = ""
    schema_text: str = ""
    error: Optional[str] = None


class GraphSearcher:
    def __init__(self, uri: str, user: str, password: str, generator: LLMCypherGenerator):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._schema = SchemaIntrospector(self._driver)
        self._generator = generator

    def close(self) -> None:
        self._driver.close()

    def search_streaming(
        self,
        query: str,
        intent: Optional[str] = None,
        term: Optional[str] = None,
        resolved_terms: Optional[dict] = None,
        top_k: int = 10,
        think: bool = True,
        vector_hints: Optional[list[dict]] = None,
        spec_refs: Optional[list[str]] = None,
        model: Optional[str] = None,
        num_ctx: Optional[int] = None,
        value_concepts: Optional[list[str]] = None,
    ) -> Iterator[dict]:
        """
        Yield progress events for the graph step. Caller (orchestrator) forwards them
        into the SSE stream.

        Event shapes:
          {stage: 'graph_start',             data: {input}}
          {stage: 'graph_cypher_thinking',   data: {token, accumulated}}   # reasoning models
          {stage: 'graph_cypher_token',      data: {token, accumulated}}
          {stage: 'retrieval_graph',         data: {input, output, count, top, cypher, ...}}
        """
        schema_text = self._schema.as_text()

        # Single-model run: Cypher gen uses the same model as the answer stage
        # by default, falling back to the generator's construction-time default
        # only when the caller didn't specify (e.g. legacy tests).
        cypher_model = model or self._generator._model
        input_payload = {
            "query": query,
            "intent": intent,
            "primary_term": term,
            "resolved_terms": resolved_terms or {},
            "schema": schema_text,
            "model": cypher_model,
            "top_k": top_k,
            "vector_hints": vector_hints or [],
            "spec_refs": spec_refs or [],
        }
        yield {"stage": "graph_start", "data": {"input": input_payload}}

        # Stream the Cypher generation token-by-token
        cypher = ""
        cypher_raw = ""
        prompt = ""
        gen_error: Optional[str] = None

        # Cypher generation is template-matching against explicit pattern rules in
        # the prompt (pick A1/A2/B/D/C, fill in slots) — not a task that benefits
        # from chain-of-thought. Empirically, thinking models (qwen3:14b) burn
        # their entire num_predict budget (CYPHER_MAX_TOKENS) inside <think>,
        # reasoning circularly about schema/pattern choice and never emitting the
        # actual query — validated live: multi-term "procedure" questions produced
        # 1000+ thinking-token events and 0 Cypher tokens, failing with "LLM
        # produced empty Cypher" regardless of prompt size or num_ctx. Force
        # think=False here (independent of the caller's `think` flag, which still
        # governs the answer-generation stage where reasoning genuinely helps).
        for event in self._generator.generate_stream(
            question=query,
            schema_text=schema_text,
            intent=intent,
            resolved_terms=resolved_terms,
            primary_term=term,
            think=False,
            vector_hints=vector_hints,
            spec_refs=spec_refs,
            value_concepts=value_concepts,
            model=cypher_model,
            num_ctx=num_ctx,
        ):
            kind = event["kind"]
            if kind == "prompt":
                prompt = event["prompt"]
                # Mutate input_payload so the final retrieval_graph event carries
                # the actual prompt that was sent to the LLM (UI shows it in Input)
                input_payload["prompt"] = prompt
                continue
            if kind == "thinking":
                yield {"stage": "graph_cypher_thinking", "data": {
                    "token": event["token"],
                    "accumulated": event["accumulated"],
                }}
                continue
            if kind == "token":
                yield {"stage": "graph_cypher_token", "data": {
                    "token": event["token"],
                    "accumulated": event["accumulated"],
                }}
                continue
            if kind == "done":
                cypher = event["cypher"]
                cypher_raw = event["raw"]
                continue
            if kind == "error":
                gen_error = event["error"]
                cypher_raw = event.get("raw", "")
                break

        # If generation failed, emit final event with error and stop
        if gen_error is not None:
            yield {"stage": "retrieval_graph", "data": {
                "input": input_payload,
                "cypher": cypher,
                "cypher_raw": cypher_raw,
                "output": {"count": 0, "top": [], "error": gen_error},
                "count": 0,
                "top": [],
                "error": gen_error,
            }}
            return

        # Run the validated Cypher
        chunks: list[dict] = []
        runtime_error: Optional[str] = None
        try:
            with self._driver.session(default_access_mode="READ") as session:
                params = {"top_k": top_k}
                if term is not None:
                    params["term"] = term
                records = session.run(cypher, **params)
                raw_chunks = [dict(r) for r in records]
                chunks = [_normalize_chunk(c, i) for i, c in enumerate(raw_chunks)]
        except Exception as e:
            runtime_error = f"cypher_runtime: {type(e).__name__}: {e}"

        # Build compact preview list (top 3) — small enough to inline in SSE event
        preview = [{
            # chunk_id is what makes a saved trace analysable: without it the graph
            # stage shows WHICH spec came back but not whether the gold chunk did, so
            # "did Cypher miss it or did the reranker drop it?" needs a fresh live run.
            # (orchestrator._chunks_to_preview carries the same field — this branch
            # builds its preview inline and does not go through that helper.)
            "chunk_id": c.get("chunk_id"),
            "spec_id": c.get("spec_id", "?"),
            "section": str(c.get("section", "?")),
            "score": c.get("score"),
            "preview": (c.get("content") or "")[:100],
        } for c in chunks[:3]]

        yield {"stage": "retrieval_graph", "data": {
            "input": input_payload,
            # The Cypher itself is the *generated artifact* — neither pure input
            # nor pure output. UI surfaces it as its own panel.
            "cypher": cypher,
            "cypher_raw": cypher_raw,
            "output": {
                "count": len(chunks),
                "top": preview,
                "error": runtime_error,
            },
            # Flat fields kept for the trail summary line
            "count": len(chunks),
            "top": preview,
            "error": runtime_error,
            # Internal: the orchestrator pulls full chunks from this event
            "_chunks": chunks,
        }}

    def search(
        self,
        query: str,
        intent: Optional[str] = None,
        term: Optional[str] = None,
        resolved_terms: Optional[dict] = None,
        top_k: int = 10,
    ) -> GraphSearchResult:
        """Non-streaming convenience wrapper — collects the streaming events
        and returns a final GraphSearchResult. Kept for callers that don't want
        progress events.
        """
        chunks: list[dict] = []
        cypher = ""
        cypher_raw = ""
        prompt = ""
        model = self._generator._model
        schema_text = ""
        error: Optional[str] = None

        for event in self.search_streaming(query, intent, term, resolved_terms, top_k):
            stage = event.get("stage")
            data = event.get("data", {})
            if stage == "graph_start":
                schema_text = (data.get("input") or {}).get("schema", "")
            elif stage == "retrieval_graph":
                chunks = data.get("_chunks", []) or []
                output = data.get("output", {}) or {}
                cypher = output.get("cypher", "")
                cypher_raw = output.get("cypher_raw", "")
                error = output.get("error")
        return GraphSearchResult(
            chunks=chunks,
            cypher=cypher,
            cypher_raw=cypher_raw,
            prompt=prompt,
            model=model,
            schema_text=schema_text,
            error=error,
        )
