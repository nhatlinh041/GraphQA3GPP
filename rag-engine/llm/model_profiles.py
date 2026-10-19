"""
Per-model profiles — OOP abstraction to customize generation behavior (generation
options + prompt format) per model family. Add a new model = add one `ModelProfile`
subclass.

Why needed: Ollama models differ in (1) default `num_ctx`, (2) whether they have a
`thinking` channel, (3) sometimes needing a custom prompt/system format. Classic
example: gemma default `num_ctx≈4096` → context of 6 chunks (~6k tokens) OVERFLOWS
the window → EMPTY output. qwen3's default is large enough so no error. These
differences used to be scattered (`_THINKING_MODEL_PREFIXES` in ollama_client); now
consolidated in one place and extended via OOP.

OllamaClient calls `get_profile(model)` then applies:
  - `profile.options(num_predict=...)`  → `options` sent to Ollama (num_ctx, num_predict)
  - `profile.supports_thinking`         → whether to send the `think` flag
  - `profile.format_prompt(prompt)`     → custom prompt hook (default: unchanged)
"""
from __future__ import annotations

import os

# The single num_ctx used across a whole request when nothing more specific applies.
# MUST always end up in the `options` sent to Ollama: Ollama keys a loaded runner by
# (model, options), so a call that omits num_ctx silently gets Ollama's own 4096
# default and forces a full unload+reload of the model when the next call asks for
# 8192. Env override exists so a benchmark run can pin a different value globally.
#
# 16384 (was 8192, then 32768): the Cypher-gen prompt alone is ~25 KB (~6.4k tokens)
# and a live-traced answer prompt reached 27,343 chars (~7.4k tokens) — at 8192 the
# grounding rules sat right at the ceiling and the pattern menu risked truncation.
# 16384 clears that measured ceiling with room to spare while matching what the paper
# benchmark pins (run_bench.CONTEXT_LENGTH), so served answers and measured answers
# no longer run at different context sizes.
DEFAULT_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))


def resolve_num_ctx(model: str | None, requested: int | None = None) -> int:
    """Resolve the ONE num_ctx a request will use, at the very start of the flow.
    Priority: explicit request (UI / API) > model profile default > DEFAULT_NUM_CTX.
    Always returns a concrete int so no downstream call site can leave it unset."""
    if requested is not None:
        return requested
    return get_profile(model).num_ctx or DEFAULT_NUM_CTX


class ModelProfile:
    """Base profile — default behavior (unchanged prompt, no thinking, no num_ctx
    override). Subclass via `prefixes` (matched against `model.lower().startswith`)."""

    prefixes: tuple[str, ...] = ()
    supports_thinking: bool = False
    # None = no override (let Ollama use the model's default). Set a number when the
    # model's default is too small for long prompts.
    num_ctx: int | None = None
    num_predict: int | None = None  # default output cap (None = no cap)

    def options(self, num_predict: int | None = None, num_ctx: int | None = None) -> dict:
        """Build the `options` dict for Ollama. A passed-in `num_predict` (e.g. Cypher
        cap) overrides the profile default. A passed-in `num_ctx` (e.g. user-chosen in
        the UI) ALWAYS wins over the profile default — even GemmaProfile — because it's
        an explicit user choice.

        `num_ctx` is ALWAYS present in the result (falling back to the profile default
        then DEFAULT_NUM_CTX) — never leave it out and let Ollama pick, see
        DEFAULT_NUM_CTX above."""
        opts: dict = {}
        ctx = num_ctx if num_ctx is not None else self.num_ctx
        opts["num_ctx"] = ctx if ctx is not None else DEFAULT_NUM_CTX
        np = num_predict if num_predict is not None else self.num_predict
        if np is not None:
            opts["num_predict"] = np
        return opts

    def format_prompt(self, prompt: str) -> str:
        """Custom prompt hook for this model family. Default: unchanged.
        Override to inject a system preamble, change format, etc. for a new model."""
        return prompt


class ThinkingProfile(ModelProfile):
    """Reasoning models emit a separate `thinking` channel when `think: true`.
    Non-thinking models 400 if sent the flag, so this must be gated to this family."""

    # The `-cloud` families on Ollama Pro all emit a `thinking` channel — verified by
    # calling each with think:true and checking the response carries one. Without the
    # prefix here they fall through to ModelProfile, and the UI's Thinking toggle
    # silently does nothing for them.
    prefixes = ("deepseek-r1", "qwen3", "gpt-oss", "deepseek-v3", "deepseek-v4",
                "glm-5", "kimi-k", "minimax-m")
    supports_thinking = True


class NemotronProfile(ThinkingProfile):
    """Nemotron-3, available on Ollama Cloud's free tier. A reasoning family, so it
    inherits the `think` flag; listed separately only because the prefix match cannot
    fold it into ThinkingProfile's tuple without also claiming `nemotron-*` models
    that are not reasoning ones."""

    prefixes = ("nemotron-3",)


class GemmaProfile(ModelProfile):
    """Gemma default num_ctx ≈ 4096 — OVERFLOWS with a 6-chunk context (~6k tokens) →
    EMPTY output (done_reason='length'). Verified experimentally:
      - num_ctx=16384 → fixes it (gemma3:12b & gemma4:12b answer correctly).
      - num_ctx=8192  → STILL empty even when the prompt is only ~3.6k tokens (gemma4
        quirk), and slower than 16384. → must use 16384, do NOT drop to 8192.
    Note: a large num_ctx does not fix gemma's poor Cypher generation (model limit)."""

    prefixes = ("gemma",)
    num_ctx = 16384


# Order = prefix-match priority (specific before general). DEFAULT is the fallback.
_PROFILES: list[ModelProfile] = [
    NemotronProfile(),
    ThinkingProfile(),
    GemmaProfile(),
]
_DEFAULT = ModelProfile()


def get_profile(model: str | None) -> ModelProfile:
    name = (model or "").lower()
    for p in _PROFILES:
        if any(name.startswith(px) for px in p.prefixes):
            return p
    return _DEFAULT
