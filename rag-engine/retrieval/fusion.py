"""
Result fusion: Reciprocal Rank Fusion (RRF) + cross-encoder reranking with
score blending (Phase C, second iteration).

Both stages dedup by normalised chunk_id so the KG-ingestion artefact where
`ts_23.288_4.1` and `ts_23_288_4.1` are stored as two separate nodes for the
same logical chunk doesn't burn slots in the final top_k.

Rerank blends cross-encoder logits with upstream signals (Pattern A canonical
score, vector cosine) so chunks promoted by graph anchors are not silently
demoted by a cross-encoder that prefers keyword-dense but off-topic chunks.
Canonical chunks (upstream score == 1.0 from `section_title CONTAINS full_name`)
get a SOFT floor bypass: they survive cross-encoder logits in the range
`[floor - CANONICAL_FLOOR_DELTA, floor)` (i.e. the cross-encoder is uncertain),
but strongly-negative logits below that drop them like any other chunk.

This second iteration removed the reserved-slot mechanism and tightened the
floor bypass after benchmark debug showed the original Phase C was over-firing
on Pattern B regex matches against short identifiers (`SA`, `N6`, `UE`):
multiple unrelated sections with section_title CONTAINS the identifier all
got upstream=1.0, all bypassed the floor with logit ≪ 0, and dominated the
answer context — costing 4 regressions on the 80-question benchmark.
"""
import math
import os

from models import get_reranker_model
from retrieval.rerank_bias import compute_bias


# Configurable floor on cross-encoder scores. 0.0 is the natural "model thinks
# query and chunk are unrelated" boundary for ms-marco-MiniLM style rerankers.
# Override via env if you want to keep marginally-negative chunks (e.g. for
# eval where you want to see what the model produces from weak evidence).
_DEFAULT_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "0.0"))

# Score blending weights (Phase C). Tunable via env. Defaults chosen so the
# cross-encoder remains the dominant signal but Pattern A canonical matches
# (upstream=1.0 from section_title CONTAINS full_name) get enough weight to
# beat keyword-dense but off-topic chunks.
_BLEND_ALPHA = float(os.getenv("RERANK_BLEND_ALPHA", "0.6"))       # cross-encoder sigmoid weight
_BLEND_BETA = float(os.getenv("RERANK_UPSTREAM_BETA", "0.3"))      # upstream score weight (Pattern A / vector cosine)
_BLEND_GAMMA = float(os.getenv("RERANK_CANONICAL_GAMMA", "0.15"))  # canonical match bonus
# Feature-bias weight (goal.md Part E, G1–G3). The cross-encoder logit (α) was
# found to be *reversed* on the procedure-vs-TR-bullet pair — it can't read
# chunk_type/spec-type/procedure-context. compute_bias() encodes those as a
# bounded [-1,+1] term added here so a mislabelled ranking is actually
# correctable (β·upstream alone can't, as five rounds of tuning showed). Set to
# 0.5 so a full-strength boost (real flow, on-context) beats a full-strength TR
# penalty by ~0.65 of blend range — enough to flip the reversed pair.
_BLEND_DELTA = float(os.getenv("RERANK_FEATURE_DELTA", "0.5"))
# Strict 1.0 — vector cosine can climb to 0.95+ but is never exactly 1.0,
# so this guarantees canonical bonus only fires for Pattern A/B exact matches.
_CANONICAL_THRESHOLD = float(os.getenv("RERANK_CANONICAL_THRESHOLD", "1.0"))
# Cap how negative a cross-encoder logit may be before we drop a canonical
# chunk too. The original Phase C bypassed the floor entirely for canonical
# chunks (Pattern A/B `score = 1.0`), but Pattern B regex on short identifiers
# (`SA`, `N6`, `S1`, `UE`) over-matches: the KG returns 5+ unrelated sections
# with section_title CONTAINS the identifier, all upstream=1.0, all logit ≪ 0,
# all surviving the bypass. They poisoned answer context. Cap = floor - 2.0
# means canonical chunks survive only when the cross-encoder is "uncertain"
# (logit ∈ [−2, 0)); strongly-negative logits (< −2) drop them like the rest.
_CANONICAL_FLOOR_DELTA = float(os.getenv("RERANK_CANONICAL_FLOOR_DELTA", "2.0"))
# Reference-chain hop chunks (retrieval/reference_hop.py, `via_reference_chain`)
# get their OWN floor bypass, separate from the shared canonical one above —
# measured live (Ca 1 of de_cuong/PhuLuc_B_5ca_vector_vs_fixed.md): the answer
# chunk's cross-encoder logit was ≈ -3.0, below even the canonical floor
# (-2.0). That's expected, not a fluke — these chunks are found via an authored
# REFERENCES_CHUNK citation specifically BECAUSE they're weak on query<->chunk
# semantic similarity (the cross-encoder's core signal), so tightening the
# shared canonical delta to accommodate them would risk reopening the Pattern-B
# over-match regression that delta was tuned to close. Bypassing the floor
# entirely is safe here because provenance is much stronger than a section-
# title substring match: an explicit edge with confidence>=0.7 at KG-build
# time, from a narrow, round-robin-capped seed set (never a flood). The FINAL
# ranking (blended final_score below, including the rerank_bias boost) still
# decides whether it reaches top_k — this only stops an outright floor-reject.
RERANK_HOP_FLOOR_BYPASS = os.getenv("RERANK_HOP_FLOOR_BYPASS", "1") not in ("0", "false", "")
# Step-actor chunks (`via_step_actors`) get the same treatment, for the same reason.
# A question like "which procedures have the UDM and the NRF in the same step" is a
# question ABOUT the corpus, not about a topic, so the clause that answers it — say
# GPSI-to-Subscription-Network resolution — has almost no lexical or semantic overlap
# with the question text. The cross-encoder scores exactly that overlap, so it scores
# these near zero BY CONSTRUCTION, not because they are wrong.
#
# Measured 2026-08-24 on kg_procedure KGP-010: the graph branch returned 27 candidates
# INCLUDING the gold, and the floor let 0 through — the answer ran on empty context
# while the right clause sat in the reject pile.
#
# Safe for the same reason the hop bypass is: the justification is a structural fact
# in the KG (two resolved terms co-occurring in one Step of that clause), not a
# substring match, and the caller marks only chunks it has verified against the graph.
# The final blended ranking still decides whether such a chunk reaches top_k; this
# only stops an outright floor-reject. Default ON, `RERANK_STEP_ACTOR_FLOOR_BYPASS=0`
# to ablate.
RERANK_STEP_ACTOR_FLOOR_BYPASS = os.getenv(
    "RERANK_STEP_ACTOR_FLOOR_BYPASS", "1") not in ("0", "false", "")
# Soft minimum on chunks reaching the answer LLM. When the floor logic above
# drops everything (cross-encoder uniformly pessimistic), the answer LLM runs
# with empty context — the smoke test on 500question_qwen3_v2 showed this in
# 217/400 questions. Setting RERANK_MIN_KEEP > 0 guarantees at least N chunks
# survive: passed-floor first, then best-by-final_score from the rejects.
# Default 0 preserves the strict Phase C behavior; raise via env to trade
# noise tolerance for higher recall.
_RERANK_MIN_KEEP = int(os.getenv("RERANK_MIN_KEEP", "0"))
# Logit band for the min_keep floor. The floor ensures the answer LLM never runs
# on empty context, BUT must not revive a chunk the cross-encoder is CONFIDENT is
# off-topic — that once forced off-topic 3GPP chunks into out-of-domain (non-3GPP)
# questions and cost −6pp on Research. Measured: cross-encoder logits separate
# cleanly — in-domain graph_count=0 questions have best-chunk logit ≥ +0.33, while
# out-of-domain median ≈ −5. Only revive chunks with logit >= floor - delta
# (default -2.0): keeps in-domain, drops ~75% of OOD.
_MINKEEP_REVIVE_FLOOR_DELTA = float(os.getenv("RERANK_MINKEEP_REVIVE_DELTA", "2.0"))


# Document-side augmentation for cross-encoder pairing. 3GPP chunk content
# often uses an abbreviation that the section_title spells out in full
# (e.g. content says "The SCP has interfaces..." while the title is
# "Requirements on the Service Communication Proxy (SCP)"). Without the
# title, the cross-encoder sees only "SCP" and misses queries phrased with
# the full name. Inspired by Anthropic's Contextual Retrieval (Sep 2024) but
# free — the section_title is already authored, no LLM call needed.
def _augment_for_rerank(c: dict) -> str:
    title = (c.get("section") or c.get("section_title") or "").strip()
    content = (c.get("content") or "").strip()
    # Skip prepend when the title is already a substring of the content
    # (some short chunks duplicate the title in the body) — avoids feeding
    # the cross-encoder the same phrase twice.
    if not title or title in content:
        return content
    return f"{title}\n\n{content}"


# Query-side expansion: append validated abbreviations + full names from
# the KG so the cross-encoder can match chunks whether they use the abbrev
# or the full term. ONLY uses entries that survived TermIndex hard-validation
# upstream — never injects synonyms from training knowledge.
# The ONE place a KG-derived signal reaches the text-only baselines: `resolved`
# comes from TermIndex (MATCH (t:Term) at startup), so `bm25` and `vector_only`
# quietly get an abbreviation -> full-name synonym dictionary the corpus itself
# defines. Arguably fair (a serious sparse baseline expands synonyms too) and
# arguably not (it is still KG-supplied). TERM_EXPANSION_OFF=1 removes it so the
# question can be settled by measurement instead of argued.
# Default is OFF-the-flag / expansion ON, i.e. behaviour is unchanged.
# Kept as the module-level default; the live value is per-request (pipeline.ablations),
# so the UI and run_bench can flip it without a new server process.
TERM_EXPANSION_OFF = os.getenv("TERM_EXPANSION_OFF", "0") in ("1", "true", "True")


def _expand_query_for_rerank(query: str, resolved: dict | None) -> str:
    from pipeline import ablations as ABL   # local import: fusion must stay importable alone
    if ABL.term_expansion_off() or not resolved:
        return query
    extras: list[str] = []
    q_upper = query.upper()
    q_lower = query.lower()
    for abbrev, info in resolved.items():
        if not isinstance(info, dict):
            continue
        canon = (abbrev or "").strip()
        full = (info.get("full_name") or "").strip()
        # Only append the form that's MISSING from the original query so
        # we don't double-weight terms the user already typed.
        if canon and canon.upper() not in q_upper:
            extras.append(canon)
        if full and full.lower() not in q_lower:
            extras.append(full)
    if not extras:
        return query
    return f"{query} {' '.join(extras)}"


# Numerically stable sigmoid for cross-encoder logits in [-30, 30] range.
def _sigmoid(x: float) -> float:
    if x > 30.0:
        return 1.0
    if x < -30.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


# Normalise a chunk_id so logical duplicates (KG produced `ts_23.288_4.1` and
# `ts_23_288_4.1` for the same section) collapse to one entry in dedup checks.
def _norm_id(cid):
    return (cid or "").replace(".", "_") if cid else ""


def rrf_fusion(
    lists: list[list[dict]],
    weights: list[float] | None = None,
    k: int = 60,
    top_k: int = 10,
) -> list[dict]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.
    Formula: RRF(d) = Σ wᵢ / (k + rankᵢ(d))

    Dedups by normalised chunk_id so ts_23.288 / ts_23_288 variants collapse
    to one entry — without this, both formats accumulate separate RRF scores
    and both could land in the final top_k.
    """
    if weights is None:
        weights = [1.0] * len(lists)

    scores: dict[str, float] = {}
    chunks: dict[str, dict] = {}

    for ranked_list, weight in zip(lists, weights):
        for rank, chunk in enumerate(ranked_list):
            cid = _norm_id(chunk.get("chunk_id"))
            if not cid:
                continue
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank + 1)
            if cid not in chunks:
                chunks[cid] = chunk

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    result = []
    for cid in sorted_ids[:top_k]:
        c = dict(chunks[cid])
        c["rrf_score"] = scores[cid]
        result.append(c)
    return result


def rerank(
    query: str,
    chunks: list[dict],
    top_k: int = 6,
    min_score: float | None = None,
    resolved_terms: dict | None = None,
    min_keep: int | None = None,
) -> list[dict]:
    """
    Cross-encoder reranking with dedup + negative-score filter. Used by both
    fixed mode and react_agent fallback (when there's only 0-1 gap to fan out
    across).

    `resolved_terms` (the KG-validated `{abbrev: {full_name, ...}}` dict from
    intent extraction) lets the reranker bridge abbreviation/full-name
    asymmetry: query "5G SCP" gets expanded with "Service Communication Proxy"
    so the cross-encoder can match chunks whose content uses either form.

    `min_keep` overrides the env-level RERANK_MIN_KEEP floor for this call. The
    orchestrator raises it (e.g. to 2) when the graph branch returned nothing,
    so the best vector hits survive even if the cross-encoder is uniformly
    pessimistic — instead of the answer LLM running on empty context.
    """
    return _rerank_with_dedup(
        query, chunks, top_k, min_score=min_score,
        resolved_terms=resolved_terms, min_keep=min_keep,
    )


def rerank_per_gap(
    question: str,
    gaps: list[str],
    chunks: list[dict],
    total_top_k: int = 6,
    min_score: float | None = None,
    resolved_terms: dict | None = None,
) -> list[dict]:
    """
    Per-sub-question reranking for compound queries.

    The cross-encoder is run separately for EACH gap (sub-question) so chunks
    that focus on one specific gap aren't drowned out by chunks that match the
    full compound query superficially. We allocate slots round-robin across
    gaps, then backfill any remaining with overall reranking against the
    original question.

    Dedups by normalised chunk_id so the ts_23.288 / ts_23_288 KG duplicates
    don't burn two slots. Filters chunks below `min_score` per pass — a chunk
    can only fill a gap's slot if it scores positively against that gap.

    `resolved_terms` is forwarded to every per-gap pass so abbrev/full-name
    expansion is consistent.
    """
    if not chunks:
        return []
    if not gaps or len(gaps) <= 1:
        # Fall back to overall rerank when there's nothing to fan out across.
        return _rerank_with_dedup(
            question, chunks, total_top_k,
            min_score=min_score, resolved_terms=resolved_terms,
        )

    # Per-gap rerank: each chunk gets a separate score against each gap, plus
    # an overall score against the original question (used for backfill).
    per_gap_ranked: list[list[dict]] = []
    for gap in gaps:
        scored = _rerank_with_dedup(
            gap, chunks, top_k=len(chunks),
            min_score=min_score, resolved_terms=resolved_terms,
        )
        per_gap_ranked.append(scored)
    overall_ranked = _rerank_with_dedup(
        question, chunks, top_k=len(chunks),
        min_score=min_score, resolved_terms=resolved_terms,
    )

    # Round-robin pick: take the next-best chunk from each gap in turn until
    # we fill total_top_k slots, deduping by normalised chunk_id.
    selected: list[dict] = []
    selected_keys: set[str] = set()
    cursors = [0] * len(gaps)
    while len(selected) < total_top_k:
        progressed = False
        for gi in range(len(gaps)):
            if len(selected) >= total_top_k:
                break
            ranked = per_gap_ranked[gi]
            # Advance cursor past any chunk already selected for another gap.
            while cursors[gi] < len(ranked):
                candidate = ranked[cursors[gi]]
                cursors[gi] += 1
                key = _norm_id(candidate.get("chunk_id"))
                if not key or key in selected_keys:
                    continue
                # Annotate which gap "won" this chunk for trail visibility.
                candidate["picked_for_gap"] = gaps[gi]
                selected.append(candidate)
                selected_keys.add(key)
                progressed = True
                break
        if not progressed:
            break

    # Backfill from overall ranking if we still have slots (e.g. some gap had
    # no positively-scored chunks left to contribute).
    if len(selected) < total_top_k:
        for c in overall_ranked:
            if len(selected) >= total_top_k:
                break
            key = _norm_id(c.get("chunk_id"))
            if not key or key in selected_keys:
                continue
            c["picked_for_gap"] = "(backfill)"
            selected.append(c)
            selected_keys.add(key)

    # Final ordering: by final_score (Phase C blended). Note this score is
    # whatever the LAST rerank pass left on the chunk (the overall-question
    # pass), since we mutate in place — that's intentional, gives a consistent
    # ranking for display.
    selected.sort(key=lambda c: c.get("final_score") or 0.0, reverse=True)
    return selected


# Run a single rerank pass: dedup-by-normalised-chunk_id, score with the
# cross-encoder, drop chunks below the (per-class) floor, blend the raw logit
# with upstream signals, sort by blended `final_score`.
#
# Two floors apply:
#   - Non-canonical chunks: dropped if logit < floor (default 0.0).
#   - Canonical chunks (upstream >= 1.0): dropped if logit < floor - CANONICAL_FLOOR_DELTA.
# The wider band for canonical lets a "the cross-encoder is uncertain" call
# tilt toward the graph anchor; the lower bound prevents Pattern B regex
# over-matches (multiple irrelevant sections that happen to contain a short
# identifier) from poisoning top_k.
#
# Returns NEW dict objects (shallow copies of input + rerank_score + final_score).
# Avoids mutating input chunks so multiple per-gap passes in `rerank_per_gap`
# don't cascade-overwrite each other's scores on the same dicts.
def _rerank_with_dedup(
    query: str,
    chunks: list[dict],
    top_k: int,
    min_score: float | None = None,
    resolved_terms: dict | None = None,
    min_keep: int | None = None,
) -> list[dict]:
    if not chunks:
        return []
    floor = _DEFAULT_MIN_SCORE if min_score is None else min_score
    canonical_floor = floor - _CANONICAL_FLOOR_DELTA
    # Per-call override of the env soft floor (RERANK_MIN_KEEP). Used when the
    # graph branch was empty: keep the top vector hits rather than let a
    # uniformly-pessimistic cross-encoder cut context down to [].
    effective_min_keep = _RERANK_MIN_KEEP if min_keep is None else min_keep

    # Dedup input: when two chunks share a normalised chunk_id, keep the first.
    seen: set[str] = set()
    deduped: list[dict] = []
    for c in chunks:
        k = _norm_id(c.get("chunk_id"))
        if not k or k in seen:
            continue
        seen.add(k)
        deduped.append(c)

    # Query-side expansion: append validated abbrev + full_name from the KG so
    # the cross-encoder can match chunks regardless of which form they use
    # internally. Only applies when resolved_terms has entries.
    expanded_query = _expand_query_for_rerank(query, resolved_terms)

    # Content-side augmentation: prepend section_title to chunk content so the
    # cross-encoder sees the full canonical phrasing (3GPP section titles like
    # "Requirements on the Service Communication Proxy (SCP)" carry both full
    # name and abbrev).
    model = get_reranker_model()
    pairs = [(expanded_query, _augment_for_rerank(c)) for c in deduped]
    scores = model.predict(pairs).tolist()

    # Phase C blend with tightened floor bypass. Score every chunk, then split
    # by floor so we can supplement `passed` from `dropped` when the soft
    # minimum (RERANK_MIN_KEEP) demands it.
    passed: list[dict] = []
    dropped: list[dict] = []
    for c, s in zip(deduped, scores):
        rerank_logit = float(s)
        upstream = float(c.get("score") or 0.0)
        is_canonical = upstream >= _CANONICAL_THRESHOLD
        # Per-class floor: canonical band is wider but still bounded. Reference-
        # chain hop chunks get their own (no) floor — see RERANK_HOP_FLOOR_BYPASS.
        if ((RERANK_HOP_FLOOR_BYPASS and c.get("via_reference_chain"))
                or (RERANK_STEP_ACTOR_FLOOR_BYPASS and c.get("via_step_actors"))):
            effective_floor = -math.inf
        else:
            effective_floor = canonical_floor if is_canonical else floor
        canonical = 1.0 if is_canonical else 0.0
        # Feature bias (goal.md Part E): structural promote/demote the cross-encoder
        # can't read. Uses the ORIGINAL query (not the abbrev-expanded one) so the
        # procedure-keyword match reads the user's phrasing.
        feature_bias = compute_bias(c, query, resolved_terms)
        final = (
            _BLEND_ALPHA * _sigmoid(rerank_logit)
            + _BLEND_BETA * upstream
            + _BLEND_GAMMA * canonical
            + _BLEND_DELTA * feature_bias
        )
        rec = {
            **c,
            "rerank_score": rerank_logit,  # raw logit kept for debug / observability
            "feature_bias": feature_bias,   # Part E bias, for debug / observability
            "final_score": final,           # blended score is the sort key
        }
        if rerank_logit < effective_floor:
            dropped.append(rec)
        else:
            passed.append(rec)

    # Floor bypass: when fewer than RERANK_MIN_KEEP chunks survived, top up
    # from the best-of-the-rejects (highest blended final_score among the
    # floor-failed pool) so the answer LLM never runs on empty context.
    if effective_min_keep > 0 and len(passed) < effective_min_keep and dropped:
        # Logit-gate the revival: only resurrect chunks the cross-encoder is NOT
        # confident are irrelevant (logit >= floor - delta). Excludes off-topic
        # 3GPP chunks on out-of-domain questions (logit ≪ 0) while keeping
        # in-domain hits. If nothing qualifies, context stays empty (correct for
        # out-of-domain — answer LLM falls back to parametric instead of noise).
        revive_floor = floor - _MINKEEP_REVIVE_FLOOR_DELTA
        eligible = [c for c in dropped if c["rerank_score"] >= revive_floor]
        eligible.sort(key=lambda c: c["final_score"], reverse=True)
        needed = effective_min_keep - len(passed)
        passed.extend(eligible[:needed])

    passed.sort(key=lambda c: c["final_score"], reverse=True)
    return passed[:top_k]
