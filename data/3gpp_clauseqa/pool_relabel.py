"""TREC-style pooling to test the gold labels of 3gpp_clauseqa for graph bias.

The gold of the set was authored FROM the knowledge graph (factoid / procedure
questions read off nodes; procedure gold "completed" by reading every clause
template F matches). A correct clause the graph cannot see therefore never becomes
gold, and a text retriever that finds it scores a miss. This script measures how big
that effect is, the way TREC does: pool the top-k of every configuration of the
paper run, have a judge read every pooled clause blind (existing gold included, as a
control), and recompute Recall@5 on old labels vs old ∪ newly-judged-relevant.

Stage 1  sample   : stratified 100 questions per class, seed 20260817 -> sample.json
Stage 2  pool     : union of top-POOL_DEPTH chunk_ids over POOL_CONFIGS of RUN, plus gold
Stage 3  judge    : one Ollama Cloud call per (question, chunk); resumable sidecar
Stage 4  report   : Recall@5 per config under old / strict-new / lenient-new labels

Outputs live in tests/benchmark/3gpp_clauseqa/pooling/ (never /tmp).
Cloud judge only -> no local GPU; JUDGE_CONCURRENCY threads (adapter caps at 3).
"""
import json
import os
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.append(str(BENCH))
JUDGE_BACKEND = os.getenv("JUDGE_BACKEND", "ollama_cloud")  # ollama_cloud | deepseek
if JUDGE_BACKEND == "deepseek":
    # deepseek_json sizes its HTTP pool from JUDGE_CONCURRENCY at import time.
    from deepseek_json import deepseek_json as _judge_call  # noqa: E402
    _DEFAULT_MODEL = "deepseek-flash"
else:
    from ollama_cloud_json import ollama_cloud_json as _judge_call  # noqa: E402
    _DEFAULT_MODEL = "deepseek-v4.1-flash:cloud"

RUN = os.getenv("RUN", "v54_3000qa_qwen3_01092026")
RUN_DIR = HERE / "runs" / RUN
EVAL_FILE = Path(os.getenv("EVAL_FILE", BENCH / "3gpp_clauseqa" / "3gpp_clauseqa.json"))
OUT_DIR = Path(os.getenv("OUT_DIR", BENCH / "3gpp_clauseqa" / "pooling"))
POOL_CONFIGS = os.getenv(
    "POOL_CONFIGS",
    "A_bm25_noterm,A_vector_noterm,A_kg_notitle,A_fixed,A_hybrid,"
    "A_bm25_dense_noterm,A_bm25_dense_noterm_norr").split(",")
POOL_DEPTH = int(os.getenv("POOL_DEPTH", "5"))
PER_CLASS = int(os.getenv("PER_CLASS", "100"))
SEED = 20260817
JUDGE_MODEL = os.getenv("JUDGE_MODEL", _DEFAULT_MODEL)
JUDGE_CONCURRENCY = int(os.getenv("JUDGE_CONCURRENCY", "3"))
# The 30-sample judge audit found a 4,000-char cut to be a failure mode; the cloud
# window is large, so we give the judge three times that.
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "12000"))

SAMPLE_F = OUT_DIR / "sample.json"
POOL_F = OUT_DIR / "pool.json"
JUDG_F = OUT_DIR / os.getenv("JUDG_FILE", "judgments.json")
FLUSH_EVERY = int(os.getenv("FLUSH_EVERY", "500"))
SUMMARY_F = OUT_DIR / "summary.json"

PROMPT = """You are labelling passage relevance for a retrieval benchmark over 3GPP specifications.

QUESTION:
{question}

REFERENCE ANSWER (written by the benchmark author; it names the clauses already labelled as correct, but other clauses may state the same facts):
{reference}

CANDIDATE CLAUSE
Specification: {spec}    Clause id: {chunk_id}
Title: {title}
Text{trunc}:
\"\"\"
{content}
\"\"\"

Decide whether THIS clause, read on its own, contains the information the question asks for.

Labels:
- RELEVANT: a reader could answer the question, or a distinct part of a multi-part question, from this clause's text alone. It need not be the clause the reference answer names; another specification may state the same fact.
- PARTIAL: the clause states some of the needed facts but not enough to answer the question or any distinct part of it.
- NOT_RELEVANT: the clause only mentions the entities, is a scope / abbreviation / table-of-contents clause, or describes a different procedure, message, parameter, value, or requirement.

For a question of the form "which procedures place A and B in a common step", RELEVANT means this clause describes a procedure in which A and B both act within the same step.
For a question asking where something is defined or which specification defines it, RELEVANT means this clause itself defines or specifies that thing (not merely refers to it).

Answer with JSON only: {{"label": "RELEVANT" | "PARTIAL" | "NOT_RELEVANT", "reason": "<one sentence>"}}"""


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load_questions():
    qs = json.load(open(EVAL_FILE))
    return qs if isinstance(qs, list) else qs["questions"]


def stage_sample(qs):
    """Stratified sample. If a smaller sample already exists it is EXTENDED, not
    redrawn, so every judgment made on it stays valid (judgments are keyed by
    question and chunk); the extension draws from the remainder with a derived seed."""
    sample = json.load(open(SAMPLE_F)) if SAMPLE_F.exists() else []
    have = {q["qid"] for q in sample}
    rng = random.Random(SEED if not sample else SEED + len(sample))
    grown = False
    for cls in ("factoid", "procedure", "requirement"):
        cur = [q for q in sample if q["question_class"] == cls]
        need = PER_CLASS - len(cur)
        if need > 0:
            rest = [q for q in qs if q["question_class"] == cls and q["qid"] not in have]
            sample += rng.sample(rest, need)
            grown = True
    if SAMPLE_F.exists() and not grown:
        return sample
    sample.sort(key=lambda q: (("factoid", "procedure", "requirement").index(q["question_class"]), q["qid"]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(sample, open(SAMPLE_F, "w"), indent=1, ensure_ascii=False)
    log(f"sample: {len(sample)} questions -> {SAMPLE_F}")
    return sample


def load_run_records():
    recs = {}
    for c in POOL_CONFIGS:
        m = json.load(open(RUN_DIR / f"{c}.json"))
        m = m if isinstance(m, list) else m["records"]
        recs[c] = {r["qid"]: r["chunk_ids"] for r in m}
    return recs


def stage_pool(sample, recs):
    """pool[qid] = {chunk_id: {"gold": bool, "from": [configs that ranked it <= depth]}}"""
    pool = {}
    for q in sample:
        qid = q["qid"]
        entry = {}
        for cid in q["gold_chunk_ids"]:
            entry[cid] = {"gold": True, "from": []}
        for c in POOL_CONFIGS:
            for cid in recs[c].get(qid, [])[:POOL_DEPTH]:
                entry.setdefault(cid, {"gold": False, "from": []})["from"].append(c)
        pool[qid] = entry
    json.dump(pool, open(POOL_F, "w"), indent=1)
    n = sum(len(v) for v in pool.values())
    ng = sum(1 for v in pool.values() for e in v.values() if e["gold"])
    log(f"pool: {n} (question, chunk) pairs, {ng} of them existing gold -> {POOL_F}")
    return pool


def fetch_chunks(chunk_ids):
    from dotenv import load_dotenv
    from neo4j import GraphDatabase
    load_dotenv(BENCH.parent.parent / ".env")
    drv = GraphDatabase.driver(os.environ["NEO4J_URI"],
                               auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))
    out = {}
    ids = sorted(chunk_ids)
    with drv.session() as s:
        for i in range(0, len(ids), 500):
            rows = s.run("UNWIND $ids AS id MATCH (c:Chunk {chunk_id: id}) "
                         "RETURN c.chunk_id AS id, c.spec_id AS spec, c.section_title AS title, "
                         "c.content AS content", ids=ids[i:i + 500])
            for r in rows:
                out[r["id"]] = {"spec": r["spec"], "title": r["title"] or "", "content": r["content"] or ""}
    drv.close()
    missing = [c for c in ids if c not in out]
    if missing:
        log(f"WARNING {len(missing)} chunk ids not in KG, e.g. {missing[:5]}")
    return out


def judge_one(q, cid, ch):
    content = ch["content"]
    trunc = ""
    if len(content) > MAX_CHUNK_CHARS:
        content = content[:MAX_CHUNK_CHARS]
        trunc = f" (first {MAX_CHUNK_CHARS} of {len(ch['content'])} characters)"
    prompt = PROMPT.format(question=q["question"], reference=q.get("reference_answer") or "(none)",
                           spec=ch["spec"], chunk_id=cid, title=ch["title"], content=content, trunc=trunc)
    parsed, err = _judge_call(prompt, model=JUDGE_MODEL, seed=SEED)
    label = (parsed or {}).get("label", "")
    if label not in ("RELEVANT", "PARTIAL", "NOT_RELEVANT"):
        return {"label": "ERROR", "reason": f"{err or ''} raw={json.dumps(parsed)[:200]}",
                "judge": f"{JUDGE_BACKEND}/{JUDGE_MODEL}"}
    return {"label": label, "reason": (parsed.get("reason") or "")[:400], "truncated": bool(trunc),
            "judge": f"{JUDGE_BACKEND}/{JUDGE_MODEL}"}


def stage_judge(sample, pool):
    judg = json.load(open(JUDG_F)) if JUDG_F.exists() else {}
    todo = [(q, cid) for q in sample for cid in pool[q["qid"]]
            if judg.get(f"{q['qid']}|{cid}", {}).get("label") in (None, "ERROR")]
    log(f"judge: {len(todo)} pairs to judge ({len(judg)} already done) backend={JUDGE_BACKEND} model={JUDGE_MODEL} threads={JUDGE_CONCURRENCY}")
    if not todo:
        return judg
    chunks = fetch_chunks({cid for _, cid in todo})
    lock = threading.Lock()
    done = [0]

    def work(q, cid):
        ch = chunks.get(cid)
        if ch is None:
            return q["qid"], cid, {"label": "ERROR", "reason": "chunk not in KG"}
        return q["qid"], cid, judge_one(q, cid, ch)

    with ThreadPoolExecutor(max_workers=JUDGE_CONCURRENCY) as ex:
        futs = [ex.submit(work, q, cid) for q, cid in todo]
        for f in as_completed(futs):
            qid, cid, res = f.result()
            with lock:
                judg[f"{qid}|{cid}"] = res
                done[0] += 1
                if done[0] % FLUSH_EVERY == 0 or done[0] == len(todo):
                    json.dump(judg, open(JUDG_F, "w"), indent=1, ensure_ascii=False)
                    c = Counter(v["label"] for v in judg.values())
                    log(f"  {done[0]}/{len(todo)}  {dict(c)}")
    json.dump(judg, open(JUDG_F, "w"), indent=1, ensure_ascii=False)
    return judg


def recall_at_k(ranked, gold, k=5):
    if not gold:
        return None
    return len(set(ranked[:k]) & gold) / len(gold)


def hit_at_k(ranked, gold, k=5):
    return 1.0 if set(ranked[:k]) & gold else 0.0


def stage_report(sample, pool, recs, judg):
    def lab(qid, cid):
        return judg.get(f"{qid}|{cid}", {}).get("label")

    labels = {"old": {}, "strict": {}, "lenient": {}}
    for q in sample:
        qid = q["qid"]
        g = set(q["gold_chunk_ids"])
        rel = {cid for cid in pool[qid] if lab(qid, cid) == "RELEVANT"}
        par = {cid for cid in pool[qid] if lab(qid, cid) == "PARTIAL"}
        labels["old"][qid] = g
        labels["strict"][qid] = g | rel
        labels["lenient"][qid] = g | rel | par

    by_class = defaultdict(list)
    for q in sample:
        by_class[q["question_class"]].append(q)
    by_class["all"] = list(sample)

    summary = {"run": RUN, "configs": POOL_CONFIGS, "depth": POOL_DEPTH, "judge": JUDGE_MODEL,
               "n_questions": len(sample), "per_class": {}, "gold_control": {}, "new_positives": {},
               "recall5": {}, "hit5": {}}

    # Judge agreement on existing gold (blind control).
    for cls, qs in by_class.items():
        c = Counter(lab(q["qid"], cid) for q in qs for cid in q["gold_chunk_ids"])
        summary["gold_control"][cls] = dict(c)
        # New positives: non-gold pool chunks judged RELEVANT / PARTIAL, and which branch
        # surfaced them.
        rel = [(q["qid"], cid) for q in qs for cid, e in pool[q["qid"]].items()
               if not e["gold"] and lab(q["qid"], cid) == "RELEVANT"]
        par = [(q["qid"], cid) for q in qs for cid, e in pool[q["qid"]].items()
               if not e["gold"] and lab(q["qid"], cid) == "PARTIAL"]
        src = Counter()
        for qid, cid in rel:
            for cfg in pool[qid][cid]["from"]:
                src[cfg] += 1
        q_with_new = len({qid for qid, _ in rel})
        summary["new_positives"][cls] = {
            "n_gold": sum(len(q["gold_chunk_ids"]) for q in qs),
            "n_unlabelled_judged": sum(1 for q in qs for cid, e in pool[q["qid"]].items() if not e["gold"]),
            "relevant": len(rel), "partial": len(par),
            "questions_gaining_gold": q_with_new, "surfaced_by": dict(src)}
        summary["per_class"][cls] = len(qs)

    for scheme in ("old", "strict", "lenient"):
        summary["recall5"][scheme] = {}
        summary["hit5"][scheme] = {}
        for cfg in POOL_CONFIGS:
            summary["recall5"][scheme][cfg] = {}
            summary["hit5"][scheme][cfg] = {}
            for cls, qs in by_class.items():
                vals = [recall_at_k(recs[cfg][q["qid"]], labels[scheme][q["qid"]]) for q in qs]
                hits = [hit_at_k(recs[cfg][q["qid"]], labels[scheme][q["qid"]]) for q in qs]
                summary["recall5"][scheme][cfg][cls] = round(sum(vals) / len(vals), 3)
                summary["hit5"][scheme][cfg][cls] = round(sum(hits) / len(hits), 3)

    # Which single branch gains most per class (strict − old), the quantity the critique is about.
    summary["delta_strict_minus_old"] = {
        cfg: {cls: round(summary["recall5"]["strict"][cfg][cls] - summary["recall5"]["old"][cfg][cls], 3)
              for cls in by_class} for cfg in POOL_CONFIGS}
    json.dump(summary, open(SUMMARY_F, "w"), indent=1)
    log(f"summary -> {SUMMARY_F}")

    # Console table.
    print("\nRecall@5  (old / strict-new / lenient-new)")
    for cls in ("factoid", "procedure", "requirement", "all"):
        print(f"\n[{cls}]  gold control: {summary['gold_control'][cls]}   "
              f"new RELEVANT {summary['new_positives'][cls]['relevant']} "
              f"/ PARTIAL {summary['new_positives'][cls]['partial']} "
              f"on {summary['new_positives'][cls]['n_unlabelled_judged']} unlabelled")
        for cfg in POOL_CONFIGS:
            o, s, l = (summary["recall5"][k][cfg][cls] for k in ("old", "strict", "lenient"))
            print(f"  {cfg:28s} {o:.3f}  {s:.3f} ({s - o:+.3f})  {l:.3f} ({l - o:+.3f})")
    return summary


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    qs = load_questions()
    sample = stage_sample(qs)
    recs = load_run_records()
    pool = stage_pool(sample, recs) if (stage in ("all", "pool", "judge") or not POOL_F.exists()) \
        else json.load(open(POOL_F))
    if stage in ("pool",):
        return
    judg = stage_judge(sample, pool) if stage in ("all", "judge") else json.load(open(JUDG_F))
    stage_report(sample, pool, recs, judg)


if __name__ == "__main__":
    main()
