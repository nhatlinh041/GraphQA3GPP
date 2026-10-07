"""
Ollama local LLM client — supports both single-shot and SSE streaming.

Per-model behavior (thinking channel, num_ctx, prompt format) is delegated to
`ModelProfile` (llm/model_profiles.py) — add a new model = add a subclass there,
no change here. E.g. gemma needs a larger num_ctx than its default (else the
6-chunk context overflows and the model returns an empty response).
"""
import json
import os
import random
import time
from collections.abc import Iterator
from urllib.parse import urlparse

import requests

from .model_profiles import get_profile


# Resolve Ollama base URL from env (matches predev OLLAMA_URL + old project LOCAL_LLM_URL)
def _resolve_base_url() -> str:
    raw = os.getenv("OLLAMA_URL") or os.getenv("LOCAL_LLM_URL") or "http://localhost:11434"
    # Strip trailing path like /api/chat or /api/generate, keep only scheme://host:port
    parsed = urlparse(raw)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return raw.rstrip("/")


OLLAMA_BASE_URL = _resolve_base_url()
DEFAULT_TIMEOUT = 120

# Backstop for STREAMING calls. `requests` timeout only measures the gap between
# two bytes, so a reasoning model streaming tokens continuously can run for minutes
# without the 120s ever firing (observed: one degenerate Cypher stream hit ~1.4 GB
# over ~9.6 min). The two caps below cut runaway loops by total time and total output
# chars. Generous enough for valid long answers; override via env.
STREAM_MAX_SECONDS = float(os.getenv("OLLAMA_STREAM_MAX_SECONDS", "180"))
STREAM_MAX_CHARS = int(os.getenv("OLLAMA_STREAM_MAX_CHARS", "200000"))

# Retry on 429. Ollama Cloud rate-limits under load, and without this the 429 becomes
# an exception the caller turns into an SSE `error` event that run_bench never reads —
# so the record lands with an EMPTY answer and `error: None`. Measured 2026-08-31 on
# run v51: 88-97% of answers empty across three configs, benchmark log completely
# clean, discovered only when the judge scored everything INCORRECT with the rationale
# "empty answer".
#
# Backoff is exponential with jitter: a fixed sleep makes every parallel worker retry
# in lockstep and re-trigger the same limit.
RETRY_STATUS = {429, 500, 502, 503, 504}
RETRY_MAX = int(os.getenv("OLLAMA_RETRY_MAX", "5"))
RETRY_BASE_S = float(os.getenv("OLLAMA_RETRY_BASE_S", "2.0"))

# One pooled Session for the whole process. `requests.post` at call level opens a new
# TCP connection every time, which at 20 concurrent questions costs more than the
# inference: measured 0.65 q/s unpooled vs 3.05 q/s pooled. pool_block=True so an
# overflow WAITS instead of having its connection discarded (urllib3 defaults to
# pool_block=False, which is what produced 408 ConnectionErrors in a 500-thread run).
_POOL = int(os.getenv("OLLAMA_POOL_SIZE", "64"))
_SESSION = requests.Session()
_SESSION.mount("http://", requests.adapters.HTTPAdapter(
    pool_connections=_POOL, pool_maxsize=_POOL, pool_block=True))
_SESSION.mount("https://", requests.adapters.HTTPAdapter(
    pool_connections=_POOL, pool_maxsize=_POOL, pool_block=True))


def _post_retry(url: str, *, json: dict, timeout: int, stream: bool = False):
    """POST with backoff on rate-limit / transient 5xx. Raises on the final attempt so
    the caller still sees a real error rather than a silent empty answer."""
    last = None
    for attempt in range(RETRY_MAX):
        r = _SESSION.post(url, json=json, timeout=timeout, stream=stream)
        if r.status_code not in RETRY_STATUS:
            r.raise_for_status()
            return r
        last = r
        if attempt < RETRY_MAX - 1:
            time.sleep(RETRY_BASE_S * (2 ** attempt) * (0.5 + random.random()))
    last.raise_for_status()
    return last


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self._base_url = base_url.rstrip("/")

    def generate(
        self,
        prompt: str,
        model: str,
        timeout: int = DEFAULT_TIMEOUT,
        format: str | None = None,
        think: bool = True,
        num_predict: int | None = None,
        num_ctx: int | None = None,
    ) -> str:
        """Single-shot generation — waits for full response.
        `format='json'` forces Ollama to return valid JSON (for structured output
        like intent classification). `think=False` disables chain-of-thought on
        reasoning models so we don't wait through a thinking phase before the response.
        `num_predict` caps generated tokens (Ollama `options.num_predict`) — set for
        tasks with fixed short output (e.g. Cypher) to block runaway generation.
        `num_ctx` overrides the context window (e.g. user pick on the UI) and always
        wins over the model profile default."""
        profile = get_profile(model)
        url = f"{self._base_url}/api/generate"
        payload: dict = {
            "model": model,
            "prompt": profile.format_prompt(prompt),
            "stream": False,
        }
        if format:
            payload["format"] = format
        opts = profile.options(num_predict=num_predict, num_ctx=num_ctx)
        if opts:
            payload["options"] = opts
        # Reasoning model: must send `think` explicitly (omitting it defaults to ON)
        if profile.supports_thinking:
            payload["think"] = bool(think)
        response = _post_retry(url, json=payload, timeout=timeout)
        return response.json()["response"]

    def generate_stream(
        self,
        prompt: str,
        model: str,
        timeout: int = DEFAULT_TIMEOUT,
        num_ctx: int | None = None,
    ) -> Iterator[str]:
        """Streaming generation — yields response tokens one by one (no thinking).
        `num_ctx` must match the rest of the request — see `generate`."""
        for ev in self.generate_stream_full(
            prompt, model=model, think=False, timeout=timeout, num_ctx=num_ctx
        ):
            if ev["kind"] == "response":
                yield ev["token"]

    def generate_stream_full(
        self,
        prompt: str,
        model: str,
        think: bool = True,
        timeout: int = DEFAULT_TIMEOUT,
        num_predict: int | None = None,
        num_ctx: int | None = None,
        max_seconds: float = STREAM_MAX_SECONDS,
        max_chars: int = STREAM_MAX_CHARS,
    ) -> Iterator[dict]:
        """
        Streaming generation that yields BOTH thinking and response tokens
        as separate events:
          {"kind": "thinking", "token": "..."}
          {"kind": "response", "token": "..."}
        For non-reasoning models (or think=False), only "response" events are emitted.

        `num_predict` caps tokens (Ollama `options.num_predict`). `num_ctx` overrides
        the context window (e.g. user pick on the UI), always winning over the profile
        default. `max_seconds` / `max_chars` are backstops: cut the stream when total
        time or total chars is exceeded — blocks degenerate generation running forever
        (per-read timeout can't save the stream because tokens keep flowing steadily).
        """
        profile = get_profile(model)
        url = f"{self._base_url}/api/generate"
        payload: dict = {
            "model": model,
            "prompt": profile.format_prompt(prompt),
            "stream": True,
        }
        opts = profile.options(num_predict=num_predict, num_ctx=num_ctx)
        if opts:
            payload["options"] = opts
        # For reasoning models, send `think` explicitly (true OR false) — omitting it
        # makes Ollama default to thinking ON, so we MUST send `think: false` to disable.
        # Non-reasoning models 400 on the flag, so we skip it entirely there.
        if profile.supports_thinking:
            payload["think"] = bool(think)

        start = time.monotonic()
        total_chars = 0
        # Streaming is the path the benchmark and the chat UI both take, so the retry
        # matters most here — this is where the v51 429s landed.
        with _post_retry(url, json=payload, timeout=timeout, stream=True) as response:
            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                # Thinking comes first; once response starts the model is "answering"
                think_tok = data.get("thinking") or ""
                if think_tok:
                    total_chars += len(think_tok)
                    yield {"kind": "thinking", "token": think_tok}
                resp_tok = data.get("response") or ""
                if resp_tok:
                    total_chars += len(resp_tok)
                    yield {"kind": "response", "token": resp_tok}
                if data.get("done"):
                    break
                # Backstop: stop a runaway stream before it burns minutes / GBs of
                # tokens. Exceeding either cap cuts it (closing the `with` block aborts
                # the underlying HTTP connection to Ollama).
                if total_chars >= max_chars or (time.monotonic() - start) >= max_seconds:
                    yield {
                        "kind": "response",
                        "token": "",
                        "truncated": True,
                    }
                    break

    def list_models(self) -> list[str]:
        """Return list of locally available model names."""
        url = f"{self._base_url}/api/tags"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]
