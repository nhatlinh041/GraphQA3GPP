"""
Anthropic (Claude) backend for the generation calls `OllamaClient` normally serves.

`OllamaClient` dispatches here when the model name is a Claude one (`is_claude_model`),
so no call site changes — the request body just carries `model: "claude-sonnet-5-5"`
(or the aliases `haiku` / `sonnet` / `opus`) and the key comes from ANTHROPIC_API_KEY.

Per-model thinking rules (verified against the current API reference, not assumed):
  - Opus 5.5   thinking cannot be disabled (400 at every effort) → think=False lowers
               `effort` instead.
  - Sonnet 5.5 `{"type":"disabled"}` is a 400 → think=False sends `between_tools`.
  - Haiku 5.5  `{"type":"disabled"}` is accepted (effort ≤ high).
  Sampling params (temperature/top_p/top_k) are 400 on these models, so none are sent;
  `num_ctx` (an Ollama option) has no meaning here and is ignored.
"""
from __future__ import annotations

import os
import time
from collections.abc import Iterator

# alias → API model id. A full `claude-*` id is passed through untouched.
CLAUDE_ALIASES = {
    "haiku": "claude-haiku-5-5",
    "sonnet": "claude-sonnet-5-5",
    "opus": "claude-opus-5-5",
}

DEFAULT_MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "16000"))
RETRY_MAX = int(os.getenv("ANTHROPIC_RETRY_MAX", "5"))

_JSON_ONLY = ("Respond with a single valid JSON value and nothing else — "
              "no prose, no markdown fences.")

_client = None


def is_claude_model(model: str | None) -> bool:
    name = (model or "").lower()
    return name in CLAUDE_ALIASES or name.startswith("claude-")


def resolve_model(model: str) -> str:
    return CLAUDE_ALIASES.get(model.lower(), model)


def _get_client():
    """Lazy: the key is read when first needed, so rag-engine still starts without it
    and `.env` has been loaded by then (main.py loads it before the pipeline)."""
    global _client
    if _client is None:
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set — required for Claude models")
        import anthropic  # imported here so Ollama-only deployments need not install it
        _client = anthropic.Anthropic(api_key=key, max_retries=RETRY_MAX)
    return _client


def _request_kwargs(prompt: str, model: str, think: bool, format: str | None,
                    num_predict: int | None) -> dict:
    mid = resolve_model(model)
    kw: dict = {
        "model": mid,
        "max_tokens": num_predict or DEFAULT_MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }
    if format == "json":
        kw["system"] = _JSON_ONLY
    if think:
        kw["thinking"] = {"type": "adaptive", "display": "summarized"}
    elif "opus" in mid:
        kw["output_config"] = {"effort": "low"}
    elif "sonnet" in mid:
        kw["thinking"] = {"type": "between_tools"}
    elif "haiku" in mid:
        kw["thinking"] = {"type": "disabled"}
    return kw


def generate(prompt: str, model: str, think: bool = True, format: str | None = None,
             num_predict: int | None = None) -> str:
    """Single-shot. Streams internally so a long answer never trips an HTTP timeout."""
    kw = _request_kwargs(prompt, model, think, format, num_predict)
    with _get_client().messages.stream(**kw) as stream:
        msg = stream.get_final_message()
    return "".join(b.text for b in msg.content if b.type == "text")


def generate_stream_full(prompt: str, model: str, think: bool, num_predict: int | None,
                         max_seconds: float, max_chars: int) -> Iterator[dict]:
    """Yields the same {"kind": "thinking"|"response", "token": ...} events as the
    Ollama path, including the `truncated` backstop."""
    kw = _request_kwargs(prompt, model, think, None, num_predict)
    start = time.monotonic()
    total_chars = 0
    with _get_client().messages.stream(**kw) as stream:
        for ev in stream:
            if ev.type != "content_block_delta":
                continue
            d = ev.delta
            if d.type == "thinking_delta" and d.thinking:
                total_chars += len(d.thinking)
                yield {"kind": "thinking", "token": d.thinking}
            elif d.type == "text_delta" and d.text:
                total_chars += len(d.text)
                yield {"kind": "response", "token": d.text}
            if total_chars >= max_chars or (time.monotonic() - start) >= max_seconds:
                yield {"kind": "response", "token": "", "truncated": True}
                return
        final = stream.get_final_message()
    # A safety-classifier decline is HTTP 200 with no text; surface it so the UI and a
    # benchmark record see a reason instead of a silently empty answer.
    if final.stop_reason == "refusal":
        cat = getattr(final.stop_details, "category", None)
        yield {"kind": "response", "token": f"[Claude declined this request (category: {cat})]"}
