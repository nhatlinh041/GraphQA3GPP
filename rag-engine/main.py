"""
FastAPI RAG Engine — entry point.
POST /api/query → SSE stream of stages: conversation, intent, retrieval_vector, retrieval_graph, rerank, answer, sources.
POST /api/cypher → execute read-only Cypher against Neo4j (for the Cypher Tester demo page).
GET/POST/DELETE /api/conversations[/{cid}] → chat history persistence (SQLite).
"""
import asyncio
import json
import os
import re
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncIterator

# Snapshot the REAL process environment before .env can pollute it. Retrieval
# tuning knobs (TOP_K_*, VECTOR_RESERVED_SLOTS, NEIGHBOR_MAX_EXTRA) are read from
# this snapshot by pipeline/orchestrator.py::_tune, so they can only be set on the
# command line of the run being measured — never from .env, which is shared state
# an unrelated edit can change under an experiment.
import os as _os
_PROC_ENV = dict(_os.environ)

# Load demo/.env before pipeline imports so OLLAMA_URL/NEO4J_* are available
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from neo4j import GraphDatabase
from neo4j.graph import Node, Relationship, Path as GraphPath
from neo4j.time import Date, DateTime, Time, Duration
from pydantic import BaseModel

from pipeline.orchestrator import RAGOrchestrator
from storage import ConversationStore, init_db


# Lifespan: run `init_db()` once at startup — idempotent CREATE TABLE IF NOT EXISTS.
# Uses async context manager (FastAPI ≥ 0.95 standard) instead of the deprecated `@app.on_event("startup")`.
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="3GPP RAG Engine", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = RAGOrchestrator()
# Singleton — `ConversationStore` holds no state, just a namespace for CRUD methods
_store = ConversationStore()


class QueryRequest(BaseModel):
    question: str
    # "fixed" | "react_agent" | "llm_only" | "vector_only" | "bm25" | "bm25_dense" | "kg_only" | "hybrid"
    mode: str = "fixed"
    # Retrieval ablations, per request. None = inherit the server's environment.
    term_expansion_off: bool | None = None
    rerank_off: bool | None = None
    title_search_off: bool | None = None
    kg_ablation: str | None = None      # entity|relation|hierarchy|xref|provenance|off
    kg_evidence: str | None = None      # text|triples
    provenance_on: bool | None = None
    # Require an answer: drop the abstain paths (grounding Rule 3, scope-check notice,
    # no-retrieval canned reply). For forced-choice benchmarks like TeleQnA.
    force_answer: bool = False
    model: str = "qwen3:14b"
    # When False, reasoning models skip the <think> phase (faster, no chain-of-thought stream)
    think: bool = True
    # Ollama num_ctx override chosen on the UI (8192-32768). None = model profile
    # default (resolve_num_ctx: explicit request > profile default > DEFAULT_NUM_CTX).
    # MUST stay Optional: GemmaProfile needs num_ctx=16384 (its default of ~4096
    # overflows on a multi-chunk RAG prompt -> empty response, done_reason='length').
    # A non-Optional field with a plain-int default (e.g. 8k) defeats that profile
    # override for every caller that omits the field — regressed gemma3/gemma4 to
    # ~50-60% empty answers in a 2026-07-26 6-model benchmark before being reverted.
    # Still safe for the "one num_ctx per request" invariant: resolve_num_ctx() runs
    # ONCE at orchestrator.query() entry and the resolved int is threaded through
    # every downstream Ollama call in the request, None or not.
    context_length: int | None = None


# Heartbeat interval for the SSE stream. Cloudflare tunnels (and many reverse
# proxies) close idle HTTP responses after ~30-100s; long ReAct runs have silent
# gaps (waiting on Neo4j, rerank, model warm-up) long enough to hit the threshold.
# We send an SSE comment line `: ping\n\n` every HEARTBEAT_INTERVAL_S seconds — the
# EventSource client ignores it, but the tunnel/proxy sees bytes flowing and keeps
# the connection open.
HEARTBEAT_INTERVAL_S = 10.0


async def event_stream(request: QueryRequest) -> AsyncIterator[str]:
    """Convert orchestrator events to SSE format with periodic heartbeats so
    long-running streams survive reverse-proxy idle timeouts. The producer task
    runs independently so heartbeats can be injected into silent gaps (Neo4j,
    rerank, model load) — `asyncio.wait_for` on the iterator directly is unsafe
    because it would cancel the generator mid-flight on timeout."""
    queue: asyncio.Queue = asyncio.Queue()
    sentinel = object()

    # Producer: drain orchestrator into queue; finally push sentinel or exception
    async def producer() -> None:
        try:
            async for event in orchestrator.query(
                request.question,
                request.mode,
                request.model,
                think=request.think,
                num_ctx=request.context_length,
                term_expansion_off=request.term_expansion_off,
                rerank_off=request.rerank_off,
                title_search_off=request.title_search_off,
                kg_ablation=request.kg_ablation,
                kg_evidence=request.kg_evidence,
                provenance_on=request.provenance_on,
                force_answer=request.force_answer,
            ):
                await queue.put(event)
        except Exception as exc:  # noqa: BLE001 — surface to consumer below
            await queue.put(exc)
        else:
            await queue.put(sentinel)

    task = asyncio.create_task(producer())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_INTERVAL_S)
            except asyncio.TimeoutError:
                # Silent too long → send keep-alive comment; client ignores this line
                yield ": ping\n\n"
                continue
            if item is sentinel:
                break
            if isinstance(item, Exception):
                # Surface orchestrator errors as an SSE event for the frontend to display
                yield f"data: {json.dumps({'stage': 'error', 'data': str(item)})}\n\n"
                break
            yield f"data: {json.dumps(item)}\n\n"
            await asyncio.sleep(0)
    finally:
        # Client disconnects mid-stream (cloudflared cancel, browser close) → cancel producer
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


@app.post("/api/query")
async def query(request: QueryRequest) -> StreamingResponse:
    """Main RAG query endpoint — returns SSE stream."""
    return StreamingResponse(
        event_stream(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ----- Cypher Tester endpoint -------------------------------------------------
# Shared driver for the tester page (kept separate from the orchestrator's driver
# so its lifecycle is independent and a long-running query won't stall the RAG path)
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
_cypher_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Block any clause that mutates the graph — the tester is read-only by design
_WRITE_CLAUSES = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH\s+DELETE|SET|REMOVE|DROP|"
    r"CREATE\s+CONSTRAINT|CREATE\s+INDEX|DROP\s+CONSTRAINT|DROP\s+INDEX|"
    r"LOAD\s+CSV|CALL\s+\{[^}]*\b(CREATE|MERGE|DELETE|SET|REMOVE)\b)",
    re.IGNORECASE,
)
DEFAULT_ROW_LIMIT = 200


class CypherRequest(BaseModel):
    query: str
    # Cap rows returned to avoid flooding the UI when the query has no LIMIT
    limit: int = DEFAULT_ROW_LIMIT
    # Cypher parameters bound by name (e.g. {"top_k": 10, "term": "AMF"})
    params: dict[str, Any] = {}


def _serialize_value(value: Any) -> Any:
    """Convert Neo4j-native types to JSON-friendly shapes for the UI."""
    if isinstance(value, Node):
        return {
            "_type": "node",
            "id": value.element_id,
            "labels": list(value.labels),
            "properties": {k: _serialize_value(v) for k, v in dict(value).items()},
        }
    if isinstance(value, Relationship):
        return {
            "_type": "relationship",
            "id": value.element_id,
            "type": value.type,
            "start": value.start_node.element_id if value.start_node else None,
            "end": value.end_node.element_id if value.end_node else None,
            "properties": {k: _serialize_value(v) for k, v in dict(value).items()},
        }
    if isinstance(value, GraphPath):
        return {
            "_type": "path",
            "nodes": [_serialize_value(n) for n in value.nodes],
            "relationships": [_serialize_value(r) for r in value.relationships],
        }
    if isinstance(value, (Date, DateTime, Time, Duration)):
        return str(value)
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    return value


@app.post("/api/cypher")
async def run_cypher(request: CypherRequest) -> dict:
    """Execute a read-only Cypher query and return columns + rows."""
    query = (request.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is empty")

    # Reject any write/DDL clause before sending the query to Neo4j
    if _WRITE_CLAUSES.search(query):
        raise HTTPException(
            status_code=400,
            detail="Only read-only queries are allowed (no CREATE/MERGE/DELETE/SET/REMOVE/DROP/LOAD CSV)",
        )

    limit = max(1, min(request.limit, 1000))
    started = time.perf_counter()
    try:
        with _cypher_driver.session() as session:
            result = session.run(query, **(request.params or {}))
            columns = list(result.keys()) if result.keys() else []
            rows: list[dict] = []
            for record in result:
                if len(rows) >= limit:
                    break
                rows.append({k: _serialize_value(record[k]) for k in columns})
            # Drain so consume() reports accurate counters
            summary = result.consume()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cypher error: {exc}") from exc

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    counters = summary.counters
    return {
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "truncated": len(rows) >= limit,
        "elapsed_ms": elapsed_ms,
        "stats": {
            "nodes_created": counters.nodes_created,
            "relationships_created": counters.relationships_created,
            "properties_set": counters.properties_set,
            "labels_added": counters.labels_added,
            "contains_updates": counters.contains_updates,
        },
    }


@app.get("/api/cypher/schema")
async def cypher_schema() -> dict:
    """Return KG schema summary (node labels, relationship types, sample counts)."""
    try:
        with _cypher_driver.session() as session:
            labels = [r["label"] for r in session.run("CALL db.labels() YIELD label RETURN label ORDER BY label")]
            rel_types = [
                r["relationshipType"]
                for r in session.run(
                    "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType"
                )
            ]
            label_counts = {}
            for label in labels:
                row = session.run(f"MATCH (n:`{label}`) RETURN count(n) AS c").single()
                label_counts[label] = row["c"] if row else 0
            rel_counts = {}
            for rel in rel_types:
                row = session.run(f"MATCH ()-[r:`{rel}`]->() RETURN count(r) AS c").single()
                rel_counts[rel] = row["c"] if row else 0
            # Corpus shape, which the label counts alone do not give: a reader sees
            # :Chunk 233,604 but not that they come from 1,800 documents, nor how the
            # chunker classified them. The paper states both, so the tester should be
            # able to check both. Cheap: two grouped counts over indexed properties.
            corpus = {
                "documents": (session.run("MATCH (d:Document) RETURN count(d) AS c")
                              .single() or {"c": 0})["c"],
                "chunks": label_counts.get("Chunk", 0),
                "chunk_types": {r["t"]: r["c"] for r in session.run(
                    "MATCH (c:Chunk) WHERE c.chunk_type IS NOT NULL "
                    "RETURN c.chunk_type AS t, count(*) AS c ORDER BY c DESC")},
                "specs": (session.run("MATCH (d:Document) RETURN count(DISTINCT d.spec_id) AS c")
                          .single() or {"c": 0})["c"],
            }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Schema error: {exc}") from exc

    return {
        "labels": labels,
        "label_counts": label_counts,
        "relationship_types": rel_types,
        "relationship_counts": rel_counts,
        "corpus": corpus,
    }


# ----- Benchmark JSON viewer/editor endpoints --------------------------------
# Read-only listing + read + versioned save of the benchmark datasets/run results
# under tests/benchmark. Every save snapshots the previous content into
# <dir>/.versions/<stem>/ and writes a unified diff alongside it, so edits to
# questions/reference/answers are reversible and auditable.

BENCH_DIR = (Path(__file__).resolve().parent.parent / "tests" / "benchmark").resolve()

# The benchmark aggregation is shared with paper_bench/run_bench.py so the Excel
# and /api/runs/.../overview cannot drift apart. bench_metrics is stdlib-only by
# design — nothing here should drag numpy/torch into the API process. `append`,
# not insert(0): BENCH_DIR holds generic names (qa_metrics, faithfulness) that
# must not shadow anything on the existing path.
import sys  # noqa: E402
if str(BENCH_DIR) not in sys.path:
    sys.path.append(str(BENCH_DIR))
try:
    import bench_metrics as BM  # noqa: E402
except ModuleNotFoundError:
    # The public release ships without tests/benchmark: chat and the Cypher
    # tester work as usual, only the benchmark run viewer is unavailable.
    BM = None


def _bench_safe(target: Path) -> bool:
    try:
        t = target.resolve()
    except Exception:
        return False
    return (str(t).startswith(str(BENCH_DIR) + os.sep)
            and ".trash" not in t.parts and ".versions" not in t.parts
            and t.suffix == ".json")


def _bench_classify(data: Any):
    """(kind, n, meta): 'questions' for a {questions:[...]} set, 'run' for a list
    of answer records; None otherwise."""
    if isinstance(data, dict) and isinstance(data.get("questions"), list):
        return "questions", len(data["questions"]), (data.get("meta") or {})
    if isinstance(data, list) and data and isinstance(data[0], dict) and "answer" in data[0]:
        return "run", len(data), {}
    return None, 0, {}


# Runs currently shown to the supervisor. Both viewers filter on this: the run list
# and the question-set list, so a set cannot be listed without its run or the reverse.
# Seventeen sets and fifteen runs accumulated over the project, most superseded; a
# reader picking one at random reads numbers nobody stands behind any more.
#
# An explicit list, not "every run on disk": retired runs (v47, v46, v11, ...) are
# real measurements and would come back under any derived rule. Add a key when a run
# joins the current story; nothing else needs editing.
BENCH_RUNS = (
    "thesis_teleqna10k_05102026",  # TeleQnA 10k × 5 local models of the thesis outline (llama3/mistral/gemma3/qwen3/r1), sequential
    "v59_3000qa_gemma3_29092026",  # bộ 3000 v5, gemma3:12b LOCAL, 2 luồng — model sinh thứ 2 (thay r1)
    "teleqna10k_27092026",  # TeleQnA MCQ: gemma4:31b-cloud / qwen3:14b / gemma3:12b (renamed from teleqna10k_gemma4_27092026)
    # "v57_3000qa_r1_26092026" (deepseek-r1:14b, stopped 29/09 after 2 configs) hidden 06/10/2026:
    # not used in the paper. The run stays on disk; uncomment to browse it again.
    "v54_3000qa_qwen3_01092026",   # run chính, qwen3:14b local — đang chuyển sang bộ v5 (27/09)
    "v53_3000qa_gemma4_31082026",  # gemma4:31b-cloud — đối chứng model sinh; đang cập nhật 300 câu mới (27/09)
)


@app.get("/api/benchmarks")
async def list_benchmarks() -> list:
    """List only the official question-set files (kind='questions'). Run-result
    files under runs/ are intentionally excluded from the viewer."""
    def scan():
        out = []
        # (path, qid set) per file, used after the loop to drop strict subsets — a
        # sample or a subset carved out of a larger set is not a set of its own, and
        # picking one by accident silently evaluates 100 questions instead of 3,000.
        seen: list = []
        # Newest first, by mtime -- NOT by path. Alphabetical order buries a set
        # authored today under directories that merely start with an earlier
        # letter, and question sets are added often enough that "which one did I
        # just build?" is the common question. Same reasoning as /api/runs, which
        # had the matching bug with version numbers sorting lexically.
        for p in sorted(BENCH_DIR.rglob("*.json"),
                        key=lambda x: x.stat().st_mtime, reverse=True):
            if ".trash" in p.parts or ".versions" in p.parts or "runs" in p.parts:
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            kind, n, meta = _bench_classify(data)
            if kind != "questions":
                continue
            # Symlinks are the same set under an old name (kg_bench_2100.json ->
            # kg_bench_3000.json, kept because v50's _agg.json records that path).
            # Listing both makes one set look like two.
            if p.is_symlink():
                continue
            qs = data.get("questions") if isinstance(data, dict) else data
            ids = frozenset(q.get("qid") or q.get("id") for q in qs) if isinstance(qs, list) else frozenset()
            if ids:
                seen.append((p, ids))
            out.append({
                "path": p.relative_to(BENCH_DIR).as_posix(),
                "kind": kind, "n": n,
                "name": (meta.get("name") if isinstance(meta, dict) else None) or p.stem,
                "size": p.stat().st_size,
                "mtime": int(p.stat().st_mtime),
            })
        # Drop a set whose questions are all contained in a strictly larger set IN THE
        # SAME DIRECTORY: judge_sample_110 beside kg_bench_3000, subset_100/200 beside
        # wh_kg_v5_600. These are one-off samples, and listing them as peers lets a
        # reader measure 110 questions believing they measured 3,000.
        #
        # The same-directory guard is what keeps this from over-firing. kg_value,
        # kg_win_100, kg_procedure, kg_requirement and kg_bench_300_v2 are ALSO strict
        # subsets by qid — their questions were folded into the 3,000-question set —
        # but each is an independent set with its own runs, and hiding them would
        # remove five sets a reader may legitimately want. A directory holds one set
        # plus its samples; that is the boundary being used here.
        sub = {a.relative_to(BENCH_DIR).as_posix()
               for a, ia in seen
               for b, ib in seen
               if a != b and a.parent == b.parent and ia < ib}

        # Second filter: the viewer lists the sets currently being shown to the
        # supervisor, not everything on disk. Seventeen sets accumulated over the
        # project, most of them superseded, and a reader picking one at random reads
        # numbers nobody stands behind any more.
        #
        # Keyed on BENCH_RUNS rather than "has any run at all": several retired sets
        # (kg_win_100, kg_bench_300, kg_strength_400) do have runs and would come
        # back. Add a run key when a set joins the current story.
        # kg_bench_3000_v4_to_run is deliberately NOT listed: its 120 swapped-in
        # procedure questions were selected by measured KG results, so it favours
        # the graph and is not a set the paper reports on.
        keep: set[str] = set()
        for run in BENCH_RUNS:
            agg = RUNS_DIR / run / "_agg.json"
            if not agg.is_file():
                continue
            try:
                ef = ((json.loads(agg.read_text(encoding="utf-8")) or {})
                      .get("meta") or {}).get("eval_file")
            except Exception:
                continue
            if ef:
                keep.add(Path(ef).name)
        out = [r for r in out if r["path"].split("/")[-1] in keep]
        return [r for r in out if r["path"] not in sub]
    return await asyncio.to_thread(scan)


@app.get("/api/benchmarks/file")
async def get_benchmark_file(path: str) -> dict:
    target = BENCH_DIR / path
    if not _bench_safe(target) or not target.is_file() or "runs" in target.parts:
        raise HTTPException(status_code=400, detail="invalid benchmark path")

    def read():
        data = json.loads(target.read_text(encoding="utf-8"))
        kind, _, _ = _bench_classify(data)
        return {"path": path, "kind": kind, "content": data}
    return await asyncio.to_thread(read)


# ----- Benchmark RUN explorer -------------------------------------------------
# /api/benchmarks above deliberately hides runs/ (it lists question SETS). These
# endpoints expose the run RESULTS instead, as a run → config → question tree, so a
# single answer can be reviewed the way it looks in the chat UI. The per-question
# payload is served on demand: a 600-question config file is ~1.4 MB, far too much
# to ship just to draw a list.
RUNS_DIR = (BENCH_DIR / "paper_bench" / "runs").resolve()


def _runs_safe(target: Path) -> bool:
    try:
        t = target.resolve()
    except Exception:
        return False
    return str(t).startswith(str(RUNS_DIR) + os.sep) and ".trash" not in t.parts


def _is_config_file(p: Path) -> bool:
    """A per-config record file, as opposed to a sidecar or run-level metadata.

    The leading-underscore test matters: run-level files are `_agg.json`,
    `_provenance.json`, `_judge_controls.json`… and an exact-name exclusion list
    let `_provenance` show up in the UI as a config with n=0.

    `.partial.json` is the mid-run checkpoint (added 2026-08-29 so an interrupted
    config resumes per question instead of restarting). It holds
    `{"recs": [...], "trails": {...}}`, NOT a bare record list, so letting it
    through made /api/runs/<run> raise AttributeError on `r.get` and return 500 —
    the whole run vanished from the UI while it was still being measured.
    """
    return (not p.name.startswith("_")
            and not p.name.endswith((".scored.json", ".stages.json",
                                     ".judge.json", ".partial.json",
                                     # TeleQnA trust-judge sidecar (teleqna_trust_judge.py)
                                     ".trust.json",
                                     # latency calibration sidecar (a dict, not a
                                     # record list — same 500 as .partial.json)
                                     ".latency_cal.json")))


def _run_parent(run_dir: Path) -> Path | None:
    """Parent run whose unchanged configs this run inherits, per _provenance.json.

    A targeted re-run (e.g. only the two BM25 configs after the tokenizer fix)
    stores just those, so without this the UI shows a two-row run and there is no
    way to compare the re-measured baseline against the untouched full system.
    """
    f = run_dir / "_provenance.json"
    if not f.is_file():
        return None
    try:
        name = (json.loads(f.read_text(encoding="utf-8")) or {}).get("parent_run")
    except Exception:
        return None
    if not name:
        return None
    p = RUNS_DIR / name
    return p if (_runs_safe(p) and p.is_dir()) else None


def _run_config_files(run_dir: Path, inherit: bool = True) -> list:
    """Config record files of a run, resolving inherited ones from the parent."""
    own = {p.name: p for p in run_dir.glob("*.json") if _is_config_file(p)}
    if inherit:
        parent = _run_parent(run_dir)
        if parent:
            for p in parent.glob("*.json"):
                if _is_config_file(p) and p.name not in own:
                    own[p.name] = p
    return [own[k] for k in sorted(own)]


@lru_cache(maxsize=1)
def _judge_prompt_template() -> str | None:
    """The judge rubric, read out of judge_correctness.py rather than duplicated.

    Copying the template here would let the two drift, and the viewer would then
    show a prompt the judge never saw — worse than showing none. Cached because the
    file does not change while the server runs.
    """
    f = BENCH_DIR / "paper_bench" / "judge_correctness.py"
    try:
        src = f.read_text(encoding="utf-8")
        i = src.index('_PROMPT = """') + len('_PROMPT = """')
        return src[i:src.index('"""', i)]
    except Exception:
        return None


def _judge_sidecar(run_dir: Path, config: str) -> dict:
    """One config's judge verdicts, or {} when it has not been judged.

    Read from the sidecar rather than from `.scored.json`, deliberately: the judge
    writes `<config>.judge.json` and NEVER `.scored.json`, because assemble()
    rewrites the latter whenever a records file is newer and would discard hours of
    paid LLM calls. Reading it here also means a judge run in progress shows up as
    soon as it checkpoints (every 25 items) instead of after a re-score.

    Returns the `items` map keyed by qid, plus `_meta` under a reserved key so the
    caller can label WHICH model produced the verdicts — the run may end up judged
    by more than one over time.
    """
    f = _config_path(run_dir, config, ".judge.json")
    if f is None or not f.is_file():
        return {}
    try:
        d = json.loads(f.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    items = d.get("items") or {}
    return {"items": items, "meta": d.get("meta") or {}}


def _judge_aggregate(side: dict) -> dict:
    """Mean strict/cond/abstain over the verdicts that exist.

    `n_judged` is reported separately from the config's `n` on purpose: the judge
    runs on a subset, so presenting one denominator for both would overstate the
    sample behind the correctness number. Errored verdicts are excluded rather than
    counted as zero — a failed API call is not a wrong answer.
    """
    items = [v for v in (side.get("items") or {}).values()
             if v.get("judge_label") and not v.get("judge_error")]
    if not items:
        return {}
    strict = [v["judge_score_strict"] for v in items
              if v.get("judge_score_strict") is not None]
    cond = [v["judge_score_cond"] for v in items
            if v.get("judge_score_cond") is not None]
    labels = {}
    for v in items:
        labels[v["judge_label"]] = labels.get(v["judge_label"], 0) + 1
    out = {
        "n_judged": len(items),
        "judge_labels": labels,
        "judge_model": (side.get("meta") or {}).get("model"),
        # The rubric version travels with the model: verdicts from two rubrics are not
        # comparable, and the run header is where a reader would look to find out which
        # one produced the judge columns.
        "judge_rubric": (side.get("meta") or {}).get("prompt_version"),
        "judge_abstain": labels.get("ABSTAIN", 0) / len(items),
    }
    if strict:
        out["judge_score_strict"] = sum(strict) / len(strict)
    if cond:
        out["judge_score_cond"] = sum(cond) / len(cond)
    return out


def _judge_by_group(scored: list, side: dict) -> dict:
    """Per-group judge means, joined to the scored records by qid.

    The two live apart on purpose — group membership is a property of the QUESTION
    and sits in `.scored.json`, while verdicts sit in the judge sidecar — so the
    join happens here rather than in bench_metrics, which never sees the sidecar.

    Mirrors aggregate_by_group's membership rule exactly (form type OR class OR
    subclass OR structure tag, with `or` so an overlapping record is not counted
    twice); duplicating the predicate is deliberate, since importing it would pull
    the sidecar concept into the stdlib-only metric layer.

    `n_judged` is per group and reported beside the value: the judge may have run on
    a subset, and a group of 300 questions with 40 verdicts must not present its
    mean under the group's full n.
    """
    items = (side.get("items") or {})
    if not items:
        return {}
    out = {}
    for g in BM.GROUPS:
        vs = [items.get(s.get("qid")) for s in scored
              if not s.get("error")
              and (s.get("question_type") == g
                   or s.get("question_class") == g
                   or s.get("question_subclass") == g
                   or g in (s.get("structure_tags") or []))]
        vs = [v for v in vs if v and v.get("judge_label") and not v.get("judge_error")]
        if not vs:
            continue
        strict = [v["judge_score_strict"] for v in vs
                  if v.get("judge_score_strict") is not None]
        cond = [v["judge_score_cond"] for v in vs
                if v.get("judge_score_cond") is not None]
        cell = {"n_judged": len(vs),
                "judge_abstain": sum(1 for v in vs
                                     if v.get("judge_label") == "ABSTAIN") / len(vs)}
        if strict:
            cell["judge_score_strict"] = sum(strict) / len(strict)
        if cond:
            cell["judge_score_cond"] = sum(cond) / len(cond)
        out[g] = cell
    return out


# TeleQnA subjects, in the order the paper's per-category table uses. A TeleQnA run
# (published by tests/benchmark/teleqna_to_run.py) is scored by multiple-choice
# accuracy, not Recall/judge, so its overview carries these instead of BM.GROUPS.
TELEQNA_SUBJECTS = ("Lexicon", "Research overview", "Research publications",
                    "Standards overview", "Standards specifications")
_MCQ_KEYS = ("mcq_acc", "mcq_extract_fail")


def _trust_aggregate(run_dir: Path, config: str, scored: list) -> dict:
    """TeleQnA trust verdicts (tests/benchmark/teleqna_trust_judge.py) per group.

    `trusted` = right option AND reasoning consistent with TeleQnA's official
    explanation AND no hallucinated fact; `verifiable` = trusted AND the retrieved
    passages establish the answer. Denominator is the answers JUDGED in the group
    (`n`), which lags the answered count while the judge is still running — the
    viewer prints it beside every cell for that reason.
    """
    f = run_dir / f"{config}.trust.json"
    if not f.is_file():
        return {}
    try:
        side = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {}
    items = side.get("items") or {}
    meta = {s["qid"]: s for s in scored}
    groups = {g: (lambda x, g=g: x.get("question_type") == g) for g in TELEQNA_SUBJECTS}
    groups["overall"] = lambda x: True
    groups["rel18"] = lambda x: x.get("release_tag") == "18"
    groups["3gpp_tagged"] = lambda x: x.get("release_tag") is not None
    out = {"model": (side.get("meta") or {}).get("model"),
           "prompt_version": (side.get("meta") or {}).get("prompt_version")}
    for g, pred in groups.items():
        vs = [v for q, v in items.items()
              if q in meta and pred(meta[q]) and v.get("label") and not v.get("error")]
        if vs:
            out[g] = {"n": len(vs),
                      "trusted": sum(v["label"] == "TRUSTED" for v in vs) / len(vs),
                      "verifiable": sum(v["label"] == "TRUSTED" and v.get("evidence") == "SUPPORTED"
                                        for v in vs) / len(vs)}
    return out


def _stages_for(run_dir: Path, config: str, qid: str):
    """One question's pipeline trail: the single `.stages.json` run_bench writes, or
    the 100-question shard teleqna_to_run writes (`<config>.stages/NNN.json`) — a 10k
    run as one file is ~300 MB, re-parsed on every question opened."""
    side = _config_path(run_dir, config, ".stages.json")
    if side is not None and side.is_file():
        try:
            return json.loads(side.read_text(encoding="utf-8")).get(qid)
        except Exception:
            return None
    try:
        shard = f"{int(qid.rsplit('-', 1)[1]) // 100:03d}.json"
    except (ValueError, IndexError):
        return None
    for d in (run_dir, _run_parent(run_dir)):
        f = d / f"{config}.stages" / shard if d is not None else None
        if f is not None and f.is_file():
            try:
                return json.loads(f.read_text(encoding="utf-8")).get(qid)
            except Exception:
                return None
    return None


def _config_path(run_dir: Path, config: str, suffix: str = ".json") -> Path | None:
    """Where one config's file lives — this run, or the parent it inherits from."""
    for d in (run_dir, _run_parent(run_dir)):
        if d is None:
            continue
        p = d / f"{config}{suffix}"
        if p.is_file():
            return p
    return None


# Cross-run comparison pages, listed in the run tree beside the real runs. One page per
# question set: every BENCH_RUNS run whose configs cover EXACTLY that set's qids is
# collected config by config. Membership is checked against the set file per config
# rather than read from _agg.json: a run writes _agg.json only when it finishes, and
# v53/v54 were converted to v5 in place, so their recorded eval_file is not a reliable
# signal of which questions a given config file holds.
COMPARE_PAGES = {
    "compare_kg_bench_3000_v5": BENCH_DIR / "kg_bench_3000" / "kg_bench_3000_v5.json",
}

# Generator and serving mode, for runs whose _agg.json does not exist yet or omits
# them. Latency is comparable only between runs served the same way.
RUN_SERVING = {
    "v53_3000qa_gemma4_31082026": ("gemma4:31b-cloud", "cloud, 2 threads"),
    "v54_3000qa_qwen3_01092026": ("qwen3:14b", "local, sequential"),
    "v57_3000qa_r1_26092026": ("deepseek-r1:14b", "local, 1-2 threads"),
    "v59_3000qa_gemma3_29092026": ("gemma3:12b", "local, 2 threads"),
}

# Runs left off the comparison page (still browsable on their own). v57 stopped after
# two configs when deepseek-r1 was replaced by gemma3 (v59) — a mostly-empty column.
COMPARE_HIDE = {"v57_3000qa_r1_26092026"}

_COMPARE_MAIN = ("A_llm_only", "A_bm25_noterm", "A_vector_noterm", "A_kg_notitle",
                 "A_bm25_dense_noterm_norr", "A_bm25_dense_noterm", "A_hybrid", "A_fixed")
_COMPARE_KEYS = ("n", "recall@1", "recall@5", "recall@10", "mrr",
                 "judge_score_strict", "judge_score_cond", "judge_abstain", "n_judged",
                 "faithfulness", "lat_p50", "judge_model", "judge_rubric")

# The run the paper's tables come from; its column leads every matrix.
COMPARE_PAPER_RUN = "v54_3000qa_qwen3_01092026"

# The paper draft's claims (paper_v3/sections/05-results.tex), each a paired delta
# a - b over the same questions: (key, text, metric, a, b, class or None, expected).
# expected: "pos"/"neg" = CI must exclude 0 on that side; "zero" = CI must include 0.
# Checked per generator, so the page answers "does this claim survive a different
# generator" rather than restating one run's means.
_CLAIMS = (
    ("full_vs_text_r5", "Full system beats BM25+Dense (no rerank) on Recall@5",
     "recall@5", "A_fixed", "A_bm25_dense_noterm_norr", None, "pos"),
    ("full_vs_text_judge", "…but not on answer correctness (judge strict)",
     "judge_score_strict", "A_fixed", "A_bm25_dense_noterm_norr", None, "zero"),
    ("retrieval_needed", "Retrieval is required: Full system vs LLM-only (judge strict)",
     "judge_score_strict", "A_fixed", "A_llm_only", None, "pos"),
    ("kg_factoid", "KG beats BM25 on factoid (Recall@5)",
     "recall@5", "A_kg_notitle", "A_bm25_noterm", "factoid", "pos"),
    ("kg_procedure", "KG beats BM25 on procedure (Recall@5)",
     "recall@5", "A_kg_notitle", "A_bm25_noterm", "procedure", "pos"),
    ("kg_requirement", "KG loses to BM25 on requirement (Recall@5)",
     "recall@5", "A_kg_notitle", "A_bm25_noterm", "requirement", "neg"),
    ("rerank_text", "Reranking HURTS a text-only pair (BM25+Dense, Recall@5)",
     "recall@5", "A_bm25_dense_noterm", "A_bm25_dense_noterm_norr", None, "neg"),
    ("rerank_graph", "Reranking HELPS the pair with a graph branch (Dense+KG, Recall@5)",
     "recall@5", "A_fixed", "A_hybrid", None, "pos"),
)
_BOOT_B, _BOOT_SEED = 2000, 20260817
_COMPARE_CACHE: dict = {}
_COMPARE_GROUPS = ("factoid", "procedure", "requirement")
_SET_QIDS: dict = {}


def _set_qids(path: Path) -> set:
    mt = path.stat().st_mtime
    hit = _SET_QIDS.get(path)
    if hit and hit[0] == mt:
        return hit[1]
    d = json.loads(path.read_text(encoding="utf-8"))
    qs = d.get("questions", d) if isinstance(d, dict) else d
    ids = {q["qid"] for q in qs}
    _SET_QIDS[path] = (mt, ids)
    return ids


def _per_question(run_dir: Path, config: str, metric: str) -> dict:
    """qid -> value: Recall from .scored.json, judge scores from the judge sidecar
    (the sidecar is the source of truth for verdicts, see _judge_sidecar)."""
    if metric.startswith("judge_"):
        items = _judge_sidecar(run_dir, config).get("items") or {}
        return {q: v[metric] for q, v in items.items()
                if v.get("judge_label") and not v.get("judge_error")
                and v.get(metric) is not None}
    p = _config_path(run_dir, config, ".scored.json")
    if p is None:
        return {}
    try:
        sc = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {s["qid"]: s[metric] for s in sc
            if not s.get("error") and s.get(metric) is not None}


def _claims_for(run_dir: Path, done: set, classes: dict) -> dict:
    """Paired bootstrap of every claim this run has both configs for."""
    import numpy as np
    out, cache = {}, {}

    def vals(cfg, metric):
        if (cfg, metric) not in cache:
            cache[(cfg, metric)] = _per_question(run_dir, cfg, metric)
        return cache[(cfg, metric)]

    for key, _text, metric, a, b, cls, expected in _CLAIMS:
        if a not in done or b not in done:
            continue
        va, vb = vals(a, metric), vals(b, metric)
        qs = sorted(q for q in va.keys() & vb.keys() if cls is None or classes.get(q) == cls)
        if len(qs) < 30:
            continue
        d = np.array([va[q] - vb[q] for q in qs], dtype=float)
        rng = np.random.default_rng(_BOOT_SEED)
        boot = d[rng.integers(0, len(d), size=(_BOOT_B, len(d)))].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
        mean = float(d.mean())
        holds = (lo > 0) if expected == "pos" else (hi < 0) if expected == "neg" \
            else (lo <= 0 <= hi)
        out[key] = {"delta": mean, "lo": float(lo), "hi": float(hi),
                    "p": max(float(p), 1 / _BOOT_B), "n": len(qs),
                    "a": float(np.mean([va[q] for q in qs])),
                    "b": float(np.mean([vb[q] for q in qs])), "holds": bool(holds)}
    return out


def _compare_overview(page: str) -> dict:
    # Rebuilding reads every config file of every run and bootstraps the claims —
    # a few seconds — so reuse the result until any file in those runs changes.
    sig = tuple(sorted((str(f), f.stat().st_mtime) for name in BENCH_RUNS
                       if (RUNS_DIR / name).is_dir()
                       for f in (RUNS_DIR / name).glob("*.json")))
    hit = _COMPARE_CACHE.get(page)
    if hit and hit[0] == sig:
        return hit[1]
    res = _compare_overview_build(page)
    _COMPARE_CACHE[page] = (sig, res)
    return res


def _compare_overview_build(page: str) -> dict:
    path = COMPARE_PAGES[page]
    qids = _set_qids(path)
    d_set = json.loads(path.read_text(encoding="utf-8"))
    classes = {q["qid"]: q.get("question_class")
               for q in (d_set.get("questions", d_set) if isinstance(d_set, dict) else d_set)}
    runs, order = [], list(_COMPARE_MAIN)
    names = sorted(BENCH_RUNS, key=lambda n: n != COMPARE_PAPER_RUN)
    for name in names:
        d = RUNS_DIR / name
        if name in COMPARE_HIDE or not d.is_dir():
            continue
        on_set = set()
        for p in _run_config_files(d):
            try:
                recs = json.loads(p.read_text(encoding="utf-8"))
                if {r.get("qid") for r in recs} == qids:
                    on_set.add(p.stem)
            except Exception:
                continue
        if not on_set:
            continue
        ov = _run_overview(d, name)
        meta = ov.get("meta") or {}
        model, serving = RUN_SERVING.get(name, (None, None))
        cfgs = {}
        for c in ov["configs"]:
            if c["config"] not in on_set:
                continue
            row = {k: c.get(k) for k in _COMPARE_KEYS}
            row["groups"] = {g: (c.get("groups") or {}).get(g) for g in _COMPARE_GROUPS}
            cfgs[c["config"]] = row
            if c["config"] not in order:
                order.append(c["config"])
        # A config still running has only its checkpoint; show how far it got.
        running = {}
        for p in d.glob("*.partial.json"):
            try:
                running[p.name[:-len(".partial.json")]] = len(
                    json.loads(p.read_text(encoding="utf-8")).get("recs") or [])
            except Exception:
                pass
        runs.append({"run": name, "model": meta.get("model") or model,
                     "serving": serving, "judge_model": meta.get("judge_model"),
                     "judge_rubric": meta.get("judge_rubric"),
                     "paper": name == COMPARE_PAPER_RUN,
                     "configs": cfgs, "running": running,
                     "claims": _claims_for(d, set(cfgs), classes)})
    return {"run": page, "compare": {"set": path.name, "n_questions": len(qids),
                                     "runs": runs, "config_order": order,
                                     "groups": list(_COMPARE_GROUPS),
                                     "claims": [{"key": k, "text": t, "metric": m, "a": a,
                                                 "b": b, "class": c, "expected": e}
                                                for k, t, m, a, b, c, e in _CLAIMS],
                                     "boot": {"B": _BOOT_B, "seed": _BOOT_SEED}},
            "configs": [], "meta": {}}


@app.get("/api/runs")
async def list_runs() -> list:
    def scan():
        if not RUNS_DIR.is_dir():
            return []
        out = []
        # Newest first, by mtime — NOT by name. Run names carry a version number
        # that sorts lexically, so "v10_600_19082026" lands below "v9_600_17082026"
        # (and below v8, v7, v6...): the newest run was buried at position 6 of 11
        # while five superseded ones sat above it.
        for d in sorted(RUNS_DIR.iterdir(),
                        key=lambda x: x.stat().st_mtime if x.is_dir() else 0,
                        reverse=True):
            if not d.is_dir() or d.name not in BENCH_RUNS:
                continue
            cfgs = _run_config_files(d)
            if not cfgs:
                continue
            # A TeleQnA run records its set size in _agg.json meta; its configs differ
            # in size (full 10k vs the 3,000 Standards questions), so the first config
            # file alphabetically would report whichever scope it happens to have.
            n = 0
            try:
                n = json.loads((d / "_agg.json").read_text(encoding="utf-8")) \
                        .get("meta", {}).get("n_questions") or 0
            except Exception:
                pass
            if not n:
                try:
                    n = len(json.loads(cfgs[0].read_text(encoding="utf-8")))
                except Exception:
                    n = 0
            out.append({"run": d.name, "n_configs": len(cfgs), "n_questions": n,
                        "mtime": int(d.stat().st_mtime)})
        for page, path in COMPARE_PAGES.items():
            if path.is_file():
                out.append({"run": page, "kind": "compare", "n_configs": 0,
                            "n_questions": len(_set_qids(path)), "mtime": 0})
        return out
    return await asyncio.to_thread(scan)


def _run_overview(run_dir: Path, run: str) -> dict:
    """Per-config aggregates for one run: the numbers behind the Excel sheets,
    computed from the cached *.scored.json sidecars (no LLM, no re-scoring)."""
    if BM is None:
        raise HTTPException(status_code=404, detail="benchmark viewer not available in this build")
    # Aggregation is bench_metrics' job, not ours. This endpoint used to carry its
    # own copy — its own mean(), its own group list, its own cover_em gate — which
    # could and did disagree with the Excel in the last digit. Values are no longer
    # rounded here either; the frontend formats with toFixed(3).
    out = []
    for p_rec in _run_config_files(run_dir):
        key = p_rec.stem
        p_sc = _config_path(run_dir, key, ".scored.json")
        # Inherited rows are real measurements from the parent run, but the
        # reader must be able to tell which numbers this run actually produced.
        row = {"config": key, "inherited": p_rec.parent != run_dir,
               "source_run": p_rec.parent.name}
        # latency only exists on the raw records
        try:
            recs = json.loads(p_rec.read_text(encoding="utf-8"))
            lat = sorted(r["latency"] for r in recs if r.get("latency"))
            row["n"] = len(recs)
            row["lat_p50"] = lat[len(lat) // 2] if lat else None
            row["lat_p90"] = lat[int(len(lat) * 0.9)] if lat else None
            row["empty"] = sum(1 for r in recs if not (r.get("chunk_ids") or []))
        except Exception:
            row["n"] = 0
        sc = []
        if p_sc is not None and p_sc.is_file():
            try:
                sc = json.loads(p_sc.read_text(encoding="utf-8"))
            except Exception:
                sc = []
            row.update(BM.aggregate(sc))
            row["groups"] = BM.aggregate_by_group(sc)
            if any("mcq_acc" in x for x in sc):
                mcq = BM.aggregate(sc, keys=_MCQ_KEYS)
                row.update({k: mcq[k] for k in _MCQ_KEYS})
                row["mcq_groups"] = BM.aggregate_by_group(
                    sc, "mcq_acc", TELEQNA_SUBJECTS)
                # Accuracy on the "[3GPP Release N]"-tagged questions, the subsets
                # the 3GPP RAG papers report: Rel.18 = 780 q = Chat3GPP's split.
                for tag, only in (("rel18", lambda x: x.get("release_tag") == "18"),
                                  ("3gpp_tagged", lambda x: x.get("release_tag") is not None)):
                    a = BM.aggregate(sc, keys=("mcq_acc",), only=only)
                    row[f"mcq_{tag}"] = {"mcq_acc": a["mcq_acc"], "n": a["n"]}
                tr = _trust_aggregate(run_dir, key, sc)
                if tr:
                    row["trust"] = tr
            # KG-only (and the kgabl_* configs) decline on a fifth of the set;
            # scoring a refusal as a wrong answer is a claim the data does not
            # support, so the answered-only view ships beside the full one.
            if row.get("empty_rate"):
                row["answered_only"] = BM.aggregate(
                    sc, only=lambda s: s.get("answered"))
        # Judge verdicts live in their own sidecar and are merged HERE, not into
        # .scored.json — see _judge_sidecar. A config with no sidecar simply
        # carries no judge keys, so the UI renders "not judged" rather than a
        # zero that would read as "answered everything wrong". Outside the
        # `if p_sc` above on purpose: a config can be judged before it is scored.
        #
        # BM.aggregate lists the judge fields in MEAN_KEYS, so an unjudged
        # config comes back with them set to null. Drop those first: the client
        # must be able to tell "no verdicts" (key absent) from "verdicts that
        # averaged to something", and a null in the payload reads as the latter.
        for k_j in ("judge_score_strict", "judge_score_cond", "judge_abstain",
                    "fact_recall"):
            if row.get(k_j) is None:
                row.pop(k_j, None)
        jside = _judge_sidecar(run_dir, key)
        row.update(_judge_aggregate(jside))
        # Judge per group, folded into the SAME cells as recall@5 so the UI can
        # draw both from one place. Merged rather than assigned: the recall cell
        # already there carries the group's full `n`, which the judge subset
        # must not overwrite.
        for g, cell in _judge_by_group(sc, jside).items():
            row.setdefault("groups", {}).setdefault(g, {}).update(cell)
        out.append(row)
    # Run-level provenance (model, context length, code sha, question set) is
    # written by run_bench into _agg.json. Reading it back here rather than
    # recomputing keeps the UI header and the workbook on one source.
    meta = {}
    p_agg = run_dir / "_agg.json"
    if p_agg.is_file():
        try:
            meta = (json.loads(p_agg.read_text(encoding="utf-8")) or {}).get("meta") or {}
        except Exception:
            meta = {}
    # Judge model + rubric belong in the run header, not only per config: they say
    # what produced every judge column on screen. Read from the sidecars actually
    # present rather than from _agg.json, which records the GENERATOR — a run can be
    # re-judged with a different model without the generator changing. Collected as
    # a set so a run judged in two passes with different models says so instead of
    # showing whichever config happened to sort first.
    jm = sorted({c["judge_model"] for c in out if c.get("judge_model")})
    jr = sorted({c["judge_rubric"] for c in out if c.get("judge_rubric")})
    if jm:
        meta = {**meta, "judge_model": jm[0] if len(jm) == 1 else " / ".join(jm)}
    if jr:
        meta = {**meta, "judge_rubric": jr[0] if len(jr) == 1 else " / ".join(jr)}
    return {"run": run, "configs": out, "group_order": list(BM.GROUPS), "meta": meta,
            "teleqna_subjects": list(TELEQNA_SUBJECTS)}


# NOTE: declared BEFORE /api/runs/{run}/{config} — FastAPI matches in declaration
# order, so the generic route would otherwise swallow "overview" as a config name.
@app.get("/api/runs/{run}/overview")
async def get_run_overview(run: str) -> dict:
    if run in COMPARE_PAGES:
        return await asyncio.to_thread(_compare_overview, run)
    run_dir = RUNS_DIR / run
    if not _runs_safe(run_dir) or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="run not found")
    return await asyncio.to_thread(_run_overview, run_dir, run)


@app.get("/api/runs/{run}")
async def get_run(run: str) -> dict:
    run_dir = RUNS_DIR / run
    if not _runs_safe(run_dir) or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="run not found")

    def read():
        cfgs = []
        for p in _run_config_files(run_dir):
            try:
                recs = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            n = len(recs)
            hit = sum(1 for r in recs
                      if set(r.get("gold_chunk_ids") or []) & set((r.get("chunk_ids") or [])[:5]))
            fa = [r["faithfulness"] for r in recs if r.get("faithfulness") is not None]
            lat = sorted(r["latency"] for r in recs if r.get("latency"))
            cfgs.append({
                "config": p.stem, "n": n,
                "hit5": round(hit / n, 3) if n else None,
                "faith": round(sum(fa) / len(fa), 3) if fa else None,
                "lat_p50": lat[len(lat) // 2] if lat else None,
                "empty": sum(1 for r in recs if not (r.get("chunk_ids") or [])),
                "inherited": p.parent != run_dir,
                "source_run": p.parent.name,
                # TeleQnA: multiple-choice accuracy stands in for hit@5 in the tree.
                "mcq_acc": (round(sum(1 for r in recs if r.get("mcq_correct")) / n, 3)
                            if n and any("mcq_correct" in r for r in recs) else None),
            })
        return {"run": run, "configs": cfgs}
    return await asyncio.to_thread(read)


@app.get("/api/runs/{run}/{config}")
async def get_run_config(run: str, config: str) -> dict:
    """Light index of one config: enough to draw the question list, no answers."""
    run_dir = RUNS_DIR / run
    if not _runs_safe(run_dir):
        raise HTTPException(status_code=404, detail="run not found")
    target = _config_path(run_dir, config)
    if target is None or not _runs_safe(target):
        raise HTTPException(status_code=404, detail="config not found")

    def read():
        recs = json.loads(target.read_text(encoding="utf-8"))
        # structure_tags are frozen into the RECORDS at run time, and a re-score only
        # refreshes `.scored.json` — so a set whose labels were corrected afterwards
        # would keep filtering this list by the old tags while the sheets show the new
        # ones. Overlay the scored view when it exists rather than rewriting the
        # records: those are the raw output of that run and stay as measured.
        tags = {}
        sc = _config_path(run_dir, config, ".scored.json")
        if sc is not None and sc.is_file():
            try:
                for x in json.loads(sc.read_text(encoding="utf-8")):
                    if x.get("qid") and x.get("structure_tags") is not None:
                        tags[x["qid"]] = x["structure_tags"]
            except Exception:
                tags = {}
        items = []
        for r in recs:
            gold = set(r.get("gold_chunk_ids") or [])
            got = r.get("chunk_ids") or []
            items.append({
                "qid": r.get("qid"),
                "question": r.get("question"),
                "question_type": r.get("question_type"),
                # question_class is the EVALUATION class (factoid/procedure/requirement);
                # question_type is the SUBJECT (message, sbi_operation, ...). They are two
                # axes, not two names for one: in kg_bench_3000 the 1000 factoid questions
                # carry a mechanism as their type, so a filter offering only `type` cannot
                # select them as a class at all. The explorer nests them.
                "question_class": r.get("question_class"),
                # The third axis, and the only one that subdivides procedure and
                # requirement: their `question_type` is a single value each ("procedure",
                # "requirement"), while their subclass splits into 2 and 7 groups
                # (multi-actor flow / single clause; node_behaviour, parameter_constraint,
                # policy_constraint, ...). Sending only type would leave those two classes
                # looking like they have nothing to filter by.
                "question_subclass": r.get("question_subclass"),
                "structure_tags": tags.get(r.get("qid"), r.get("structure_tags") or []),
                "hit5": bool(gold & set(got[:5])),
                "n_chunks": len(got),
                "latency": r.get("latency"),
                "faithfulness": r.get("faithfulness"),
                "error": r.get("error"),
                "repatched": r.get("repatched"),
                # TeleQnA only: multiple-choice outcome. Absent (None) on paper_bench runs.
                "mcq_correct": r.get("mcq_correct"),
                "mcq_pred": r.get("mcq_pred"),
                "mcq_gold": r.get("mcq_gold"),
            })
        return {"run": run, "config": config, "items": items}
    return await asyncio.to_thread(read)


@app.get("/api/runs/{run}/{config}/{qid}")
async def get_run_question(run: str, config: str, qid: str) -> dict:
    """One question's full record, plus the same question under every other config
    so the answers can be compared side by side."""
    run_dir = RUNS_DIR / run
    if not _runs_safe(run_dir):
        raise HTTPException(status_code=404, detail="run not found")
    target = _config_path(run_dir, config)
    if target is None or not _runs_safe(target):
        raise HTTPException(status_code=404, detail="config not found")

    def read():
        rec = None
        siblings = []
        for p in _run_config_files(run_dir):
            try:
                recs = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            hit = next((r for r in recs if r.get("qid") == qid), None)
            if hit is None:
                continue
            if p.stem == config:
                rec = hit
            gold = set(hit.get("gold_chunk_ids") or [])
            got = hit.get("chunk_ids") or []
            siblings.append({
                "config": p.stem,
                "hit5": bool(gold & set(got[:5])),
                "n_chunks": len(got),
                "latency": hit.get("latency"),
                "faithfulness": hit.get("faithfulness"),
                "answer_chars": len(hit.get("answer") or ""),
                "inherited": p.parent != run_dir,
            })
        if rec is None:
            raise HTTPException(status_code=404, detail="qid not found in this config")
        # Pipeline trail, if this run was produced after run_bench started saving it.
        # Runs recorded earlier simply have no sidecar — the viewer says so rather than
        # pretending the retrieval steps were not interesting.
        stages = _stages_for(run_dir, config, qid)
        # This question's verdict, if it has been judged. Absent key (not null)
        # when there is none, so the client distinguishes "not judged yet" from
        # "judged and scored zero".
        jside = _judge_sidecar(run_dir, config)
        verdict = (jside.get("items") or {}).get(qid)
        # Rebuild the exact prompt the judge saw, so the viewer can show it the way
        # the retrieval trail shows the Cypher and answer prompts. The sidecar does
        # not store it (2106 x 8 copies of a fixed rubric is pure waste), but it is
        # a pure function of the rubric plus three fields, and the truncation limits
        # are part of it: a 30k-char answer really is cut at 4000 before grading, and
        # a reader comparing verdict to answer must see the same text the judge did.
        jprompt = None
        if verdict:
            # Prefer the rubric SAVED WITH THE VERDICTS over the one in the current
            # source: a rubric edit must not silently rewrite what an old verdict was
            # asked. Sidecars written before that field existed fall back to the file.
            meta = jside.get("meta") or {}
            tpl = meta.get("rubric") or _judge_prompt_template()
            if tpl:
                jprompt = tpl.format(
                    question=rec.get("question") or "",
                    reference=(rec.get("reference_answer") or "")[:meta.get("reference_chars", 4000)],
                    answer=(rec.get("answer") or "")[:meta.get("answer_chars", 4000)])
        out = {"run": run, "config": config, "record": rec,
               "siblings": siblings, "stages": stages}
        if verdict:
            out["judge"] = {**verdict,
                            "model": verdict.get("_model")
                            or (jside.get("meta") or {}).get("model"),
                            "prompt": jprompt}
        return out
    return await asyncio.to_thread(read)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# ----- Conversation history endpoints (SQLite-backed) -------------------------
# Tier 1: 1 conversation/browser, no list/sidebar. Frontend stores cid in
# localStorage["chat-conversation-id"] (~36 bytes). All DB calls wrapped in to_thread.

@app.post("/api/conversations", status_code=201)
async def create_conversation() -> dict:
    """Create an empty conversation and return the new id. Used by the "New chat" button."""
    cid = await asyncio.to_thread(_store.create)
    return {"id": cid}


@app.get("/api/conversations/{cid}")
async def get_conversation(cid: str) -> dict:
    """Return conversation metadata + all messages (JSON stages/sources parsed).
    404 if cid does not exist — frontend treats as empty history."""
    conv = await asyncio.to_thread(_store.get, cid)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@app.delete("/api/conversations/{cid}", status_code=204)
async def delete_conversation(cid: str) -> Response:
    """ON DELETE CASCADE auto-deletes messages. Idempotent: 204 even if cid does not exist."""
    await asyncio.to_thread(_store.delete, cid)
    return Response(status_code=204)
