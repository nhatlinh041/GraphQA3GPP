"""Backfill structure_tags on kg_bench_900_v3 (1806 questions carry none).

Two of the three tags have a rule that reproduces the hand-authored v2 labels
EXACTLY, and both are verified against all 300 v2 questions before anything is
written:

* cross_specification -- ">=2 distinct spec_id among the gold chunks". 100/0/0.
* cross_section -- subclass "multi-actor flow" AND n_gold > n_distinct_spec, i.e.
  at least one spec contributes 2+ clauses. 65/0/0. The naive readings all fail:
  ">=4 gold chunks" gets 47/65 and ">=2 clauses in one spec" gets 20/65, because
  both compare against an absolute number. What separates the class is the RATIO --
  evidence CLUSTERED inside a spec, rather than spread one-clause-per-spec. Despite
  its name the tag is not about clauses within a single specification: 45 of the 65
  span several specs.

multi_hop is NOT derived here. It reads as "two dependent lookups", and the v2 labels
did not encode that: within one subclass, with the wording effectively identical, the
old label tracked the gold count and nothing else --

    "Which specifications define the LOCATION REPORT message, and which node
     generates it?"        -> 2 gold, TAGGED
    "Which specifications define the NG SETUP RESPONSE message, and which node
     sends it in each?"    -> 1 gold, NOT tagged

Same two clauses, same subclass, opposite labels; and the second is the one that
actually chains ("in each"). Across all 300 v2 questions the split was exactly
gold>=2 (116/0) vs gold==1 (0/175), corrected only by excluding the 9 `information
element` questions, which have 2-3 gold across 2-3 specs like the rest.

A rule reproducing those labels was therefore rejected, twice: it would have put the
tag on 1058 of 2106 questions (50% of the set, since v3 is 59% multi-gold by design),
and it duplicates cross_section -- inside `multi-actor flow`, 65 of 77 questions carry
both and none carries cross_section alone.

The tag is now assigned by READING the question text (see
.debug_context/, "chained classification"), which is the only way to capture a property of
the question rather than of its gold. Both the v2 and the v3 labels come from that
pass, so the two sets are on one standard.

`judge_key_facts_total` measures the underlying property directly, over all 2106
questions, without depending on anyone's labelling -- judge strict falls 0.777
(2 facts) -> 0.580 (4) -> 0.438 (5+). Prefer it for any claim about difficulty; keep
multi_hop for continuity with the v2 sheets."""
import argparse, json, pathlib, sys

BENCH = pathlib.Path(__file__).resolve().parents[1] / "tests" / "benchmark"
V3 = [BENCH / "kg_bench_900_v3" / f"kg_bench_900_v3_{c}.json"
      for c in ("factoid", "procedure", "requirement")]
V2 = BENCH / "kg_bench_300_v2" / "kg_bench_300_v2.json"


def load(p):
    """v2 wraps its list under a key; v3 is a bare list. Accept both."""
    d = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(d, list):
        return d, None
    for k in ("questions", "items", "data"):
        if isinstance(d.get(k), list):
            return d[k], k
    raise SystemExit(f"{p.name}: question list not found")


def specs(q):
    """Distinct spec ids in the gold set. A chunk id is <spec>_<clause>, and the clause
    itself contains dots and letters, so split on the LAST underscore."""
    return {g.rsplit("_", 1)[0] for g in q.get("gold_chunk_ids") or []}


def is_cross_spec(q):
    return len(specs(q)) >= 2


def is_cross_section(q):
    """Clustered evidence: a multi-actor-flow question whose gold puts 2+ clauses in
    at least one spec. See the module docstring for why the ratio, not a threshold."""
    gold = q.get("gold_chunk_ids") or []
    return q.get("question_subclass") == "multi-actor flow" and len(gold) > len(specs(q))


TAGS = (("cross_specification", is_cross_spec), ("cross_section", is_cross_section))


def verify():
    """Refuse to run if a rule no longer reproduces v2 — those labels ARE the spec."""
    v2, _ = load(V2)
    ok = True
    for tag, fn_rule in TAGS:
        has = lambda q: tag in (q.get("structure_tags") or [])
        tp = sum(1 for q in v2 if fn_rule(q) and has(q))
        fp = sum(1 for q in v2 if fn_rule(q) and not has(q))
        fn = sum(1 for q in v2 if not fn_rule(q) and has(q))
        print(f"  v2 check {tag:22} correct {tp:3}, extra {fp:3}, missing {fn:3}")
        ok = ok and fp == 0 and fn == 0
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if not verify():
        sys.exit("rule no longer reproduces v2 — stopping")
    total = 0
    for p in V3:
        qs, key = load(p)
        counts = {}
        for tag, fn_rule in TAGS:
            n = 0
            for q in qs:
                if not fn_rule(q):
                    continue
                # Append only. Re-running must be a no-op, and must never disturb a
                # tag written by something else.
                tags = q.setdefault("structure_tags", [])
                if tag not in tags:
                    tags.append(tag)
                    n += 1
            counts[tag] = n
        n = sum(counts.values())
        total += n
        print(f"  {p.name:44} " + "  ".join(f"{t.split('_')[1][:4]} {c:4}" for t, c in counts.items()))
        if a.apply and n:
            tmp = p.with_suffix(".json.tmp")
            payload = qs if key is None else {**json.loads(p.read_text(encoding="utf-8")), key: qs}
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(p)
    print(f"total {total} questions" + ("" if a.apply else "  (dry run — add --apply to write)"))


if __name__ == "__main__":
    main()
