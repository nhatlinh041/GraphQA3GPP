"""
Agreement and pass-rate statistics for the human validation campaign.

stdlib only, like bench_metrics.py: main.py imports this in-process and the
offline report script reuses it, so there is exactly one implementation of
every number that reaches the paper.

Each `items` argument is a list of label lists, one inner list per question
(the labels given by its raters, in any order). Items with fewer than two
labels are ignored by the agreement functions.
"""
from __future__ import annotations

import math
from collections import Counter
from itertools import combinations


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval for a proportion k/n. None when n == 0."""
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def _multi(items: list[list[str]]) -> list[list[str]]:
    return [it for it in items if len(it) >= 2]


def percent_agreement(items: list[list[str]]) -> float | None:
    """Fraction of questions on which every rater gave the same label."""
    its = _multi(items)
    if not its:
        return None
    return round(sum(1 for it in its if len(set(it)) == 1) / len(its), 4)


def fleiss_kappa(items: list[list[str]]) -> float | None:
    """Fleiss' kappa, generalised to a varying number of raters per item
    (P_i uses that item's own n_i). Raters are treated as interchangeable,
    which is what a design with rotating pairs needs."""
    its = _multi(items)
    if not its:
        return None
    cats = sorted({l for it in its for l in it})
    total = sum(len(it) for it in its)
    p_j = {c: sum(it.count(c) for it in its) / total for c in cats}
    p_bar = 0.0
    for it in its:
        n = len(it)
        p_bar += (sum(it.count(c) ** 2 for c in cats) - n) / (n * (n - 1))
    p_bar /= len(its)
    p_e = sum(v * v for v in p_j.values())
    if p_e >= 1.0:
        return 1.0
    return round((p_bar - p_e) / (1 - p_e), 4)


def pabak(items: list[list[str]], n_categories: int) -> float | None:
    """Prevalence-adjusted bias-adjusted kappa: (k*p_o - 1)/(k - 1). Reported
    beside kappa because a set that is ~90% 'pass' pins kappa low regardless
    of how well the raters actually agree."""
    p_o = percent_agreement(items)
    if p_o is None or n_categories < 2:
        return None
    return round((n_categories * p_o - 1) / (n_categories - 1), 4)


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    """Cohen's kappa for two fixed raters over the same items."""
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    p_o = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    p_e = sum(ca[c] * cb[c] for c in set(a) | set(b)) / (n * n)
    if p_e >= 1.0:
        return 1.0
    return round((p_o - p_e) / (1 - p_e), 4)


def krippendorff_alpha(items: list[list[str]]) -> float | None:
    """Krippendorff's alpha, nominal metric, tolerating missing ratings."""
    its = _multi(items)
    if not its:
        return None
    cats = sorted({l for it in its for l in it})
    n_total = sum(len(it) for it in its)
    # Observed disagreement: within-unit pairs of differing labels.
    d_o = 0.0
    for it in its:
        m = len(it)
        cnt = Counter(it)
        pairs_diff = m * (m - 1) - sum(v * (v - 1) for v in cnt.values())
        d_o += pairs_diff / (m - 1)
    d_o /= n_total
    n_c = Counter(l for it in its for l in it)
    d_e = sum(n_c[c] * n_c[k] for c in cats for k in cats if c != k) / (n_total * (n_total - 1))
    if d_e == 0:
        return 1.0
    return round(1 - d_o / d_e, 4)


def agreement_block(items: list[list[str]], n_categories: int) -> dict:
    return {
        "n_items": len(_multi(items)),
        "raw": percent_agreement(items),
        "fleiss_kappa": fleiss_kappa(items),
        "pabak": pabak(items, n_categories),
        "krippendorff_alpha": krippendorff_alpha(items),
    }


def pairwise_cohen(by_question: dict[str, dict[int, str]]) -> list[dict]:
    """by_question: qid -> {reviewer_id: label}. Cohen's kappa for every pair
    of reviewers over the questions they both labelled."""
    reviewers = sorted({r for d in by_question.values() for r in d})
    out = []
    for r1, r2 in combinations(reviewers, 2):
        a, b = [], []
        for d in by_question.values():
            if r1 in d and r2 in d:
                a.append(d[r1])
                b.append(d[r2])
        if a:
            out.append({
                "reviewers": [r1, r2],
                "n": len(a),
                "raw": round(sum(1 for x, y in zip(a, b) if x == y) / len(a), 4),
                "cohen_kappa": cohen_kappa(a, b),
            })
    return out


def rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "p": round(k / n, 4) if n else None, "ci95": wilson(k, n)}
