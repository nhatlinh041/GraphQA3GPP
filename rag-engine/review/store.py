"""
SQLite store for the human validation campaign (expert review of 3GPP-ClauseQA).

Own database file, separate from chat.db: the two have different lifetimes
(chat history is disposable, review verdicts are paper data) and a separate file
is what gets backed up. Same conventions as storage/db.py: stdlib sqlite3, WAL,
short connections, sync functions the router runs in FastAPI's threadpool.

Everything a rater sees is SNAPSHOTTED at campaign creation — question text,
reference answer, and the gold clauses' content read from Neo4j — so a KG
rebuild or an edit to the question file after the campaign opened cannot change
the evidence a verdict was given on.
"""
from __future__ import annotations

import json
import os
import random
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from . import stats as S
from . import tables as T

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = ROOT / "tests" / "benchmark" / "human_review"
DB_PATH = Path(os.getenv("REVIEW_DB_PATH") or Path(__file__).resolve().parent.parent / "data" / "review.sqlite")

C1 = ("pass", "fail", "unsure")            # question valid?
C2 = ("full", "partial", "none")           # gold contains the answer?
C3 = ("correct", "partial", "wrong")       # reference answer correct vs gold?
VERDICT = ("keep", "fix", "discard")
PASS = {"c1": "pass", "c2": "full", "c3": "correct", "verdict": "keep"}
LABELS = {"c1": C1, "c2": C2, "c3": C3, "verdict": VERDICT}
DECISIONS = ("keep", "fix", "discard")
_JUDGE_RANK = {"NOT_RELEVANT": 0, "PARTIAL": 1, "RELEVANT": 2}

_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS campaigns (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  name          TEXT NOT NULL UNIQUE,
  sample_file   TEXT NOT NULL,
  seed          INTEGER NOT NULL,
  raters_per_q  INTEGER NOT NULL,
  status        TEXT NOT NULL DEFAULT 'open',
  created_at    INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS questions (
  campaign_id   INTEGER NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
  qid           TEXT NOT NULL,
  pos           INTEGER NOT NULL,
  batch         TEXT NOT NULL,
  question      TEXT NOT NULL,
  question_class    TEXT,
  question_subclass TEXT,
  question_type     TEXT,
  spec_series       TEXT,
  gold_chunk_ids    TEXT NOT NULL,
  reference_answer  TEXT NOT NULL,
  judge_gold        TEXT,
  PRIMARY KEY (campaign_id, qid)
);
CREATE TABLE IF NOT EXISTS gold_chunks (
  campaign_id   INTEGER NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
  chunk_id      TEXT NOT NULL,
  spec_id       TEXT,
  section_id    TEXT,
  section_title TEXT,
  content       TEXT,
  n_chars       INTEGER,
  PRIMARY KEY (campaign_id, chunk_id)
);
CREATE TABLE IF NOT EXISTS reviewers (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  name          TEXT NOT NULL UNIQUE,
  role          TEXT NOT NULL DEFAULT 'rater',
  active        INTEGER NOT NULL DEFAULT 1,
  password_hash TEXT,
  created_at    INTEGER NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_reviewers_name ON reviewers(name);
CREATE TABLE IF NOT EXISTS assignments (
  campaign_id INTEGER NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
  qid         TEXT NOT NULL,
  reviewer_id INTEGER NOT NULL REFERENCES reviewers(id),
  order_idx   INTEGER NOT NULL,
  PRIMARY KEY (campaign_id, qid, reviewer_id)
);
CREATE TABLE IF NOT EXISTS ratings (
  campaign_id INTEGER NOT NULL,
  qid         TEXT NOT NULL,
  reviewer_id INTEGER NOT NULL,
  status      TEXT NOT NULL DEFAULT 'draft',
  c1 TEXT, c1_note TEXT,
  c2 TEXT, c2_note TEXT,
  irrelevant_chunks TEXT,
  own_answer  TEXT,
  c3 TEXT,
  verdict TEXT,
  comment TEXT,
  fix_text TEXT,
  t_phase1_s REAL,
  t_phase2_s REAL,
  revealed_at INTEGER,
  revision    INTEGER NOT NULL DEFAULT 0,
  created_at  INTEGER NOT NULL,
  updated_at  INTEGER NOT NULL,
  PRIMARY KEY (campaign_id, qid, reviewer_id)
);
CREATE TABLE IF NOT EXISTS adjudications (
  campaign_id INTEGER NOT NULL,
  qid         TEXT NOT NULL,
  decision    TEXT NOT NULL,
  note        TEXT,
  decided_by  TEXT,
  created_at  INTEGER NOT NULL,
  PRIMARY KEY (campaign_id, qid)
);
CREATE TABLE IF NOT EXISTS events (
  id       INTEGER PRIMARY KEY AUTOINCREMENT,
  ts       INTEGER NOT NULL,
  reviewer TEXT,
  kind     TEXT NOT NULL,
  detail   TEXT
);
"""


def _now() -> int:
    return int(time.time())


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(_SCHEMA)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(reviewers)").fetchall()}
        if "password_hash" not in cols:  # DB created before per-rater passwords
            conn.execute("ALTER TABLE reviewers ADD COLUMN password_hash TEXT")
        ccols = {r[1] for r in conn.execute("PRAGMA table_info(campaigns)").fetchall()}
        # all_raters: every assigned rater rates every question (pilot).
        # requires_campaign_id: a rater cannot open this campaign until they have
        # completed every question assigned to them in that one.
        if "all_raters" not in ccols:
            conn.execute("ALTER TABLE campaigns ADD COLUMN all_raters INTEGER NOT NULL DEFAULT 0")
        if "requires_campaign_id" not in ccols:
            conn.execute("ALTER TABLE campaigns ADD COLUMN requires_campaign_id INTEGER")
        qcols = {r[1] for r in conn.execute("PRAGMA table_info(questions)").fetchall()}
        # in_scope = 0: the question stays in the campaign with whatever ratings it
        # already has, but is no longer assigned, queued or counted (e.g. the diag
        # batch once every rater moved to the 295-question set).
        if "in_scope" not in qcols:
            conn.execute("ALTER TABLE questions ADD COLUMN in_scope INTEGER NOT NULL DEFAULT 1")
        gcols = {r[1] for r in conn.execute("PRAGMA table_info(gold_chunks)").fetchall()}
        if "tables_json" not in gcols:
            conn.execute("ALTER TABLE gold_chunks ADD COLUMN tables_json TEXT")
        conn.commit()
    backfill_tables()


def backfill_tables() -> int:
    """Snapshot structured tables for gold chunks that predate the column."""
    with connect() as conn:
        todo = sorted({r[0] for r in conn.execute("SELECT chunk_id FROM gold_chunks WHERE tables_json IS NULL")})
    if not todo:
        return 0
    found = T.load_tables(todo)
    with connect() as conn:
        for c in todo:
            conn.execute("UPDATE gold_chunks SET tables_json = ? WHERE chunk_id = ? AND tables_json IS NULL",
                         (json.dumps(found.get(c, []), ensure_ascii=False), c))
        conn.commit()
    return len(todo)


class Locked(Exception):
    """The rater must finish a prerequisite campaign (the pilot) first."""


def _lock_state(conn: sqlite3.Connection, cid: int, rid: int) -> dict | None:
    """None when the rater may work on `cid`; otherwise what blocks them."""
    row = conn.execute("SELECT requires_campaign_id FROM campaigns WHERE id = ?", (cid,)).fetchone()
    req = row[0] if row else None
    if not req:
        return None
    name = conn.execute("SELECT name FROM campaigns WHERE id = ?", (req,)).fetchone()
    n_assigned, n_done = conn.execute(
        "SELECT COUNT(a.qid), SUM(CASE WHEN r.status = 'done' THEN 1 ELSE 0 END) FROM assignments a "
        "JOIN questions qs ON qs.campaign_id = a.campaign_id AND qs.qid = a.qid AND qs.in_scope = 1 "
        "LEFT JOIN ratings r ON r.campaign_id = a.campaign_id AND r.qid = a.qid AND r.reviewer_id = a.reviewer_id "
        "WHERE a.campaign_id = ? AND a.reviewer_id = ?", (req, rid)).fetchone()
    n_done = n_done or 0
    if n_assigned and n_done >= n_assigned:
        return None
    return {"requires_campaign_id": req, "requires_name": name[0] if name else str(req),
            "n_assigned": n_assigned, "n_left": n_assigned - n_done}


def _check_unlocked(conn: sqlite3.Connection, cid: int, rid: int) -> None:
    st = _lock_state(conn, cid, rid)
    if st:
        if not st["n_assigned"]:
            raise Locked(f"Campaign locked: you have not been assigned the pilot '{st['requires_name']}' yet — contact the administrator")
        raise Locked(f"Campaign locked: finish the pilot '{st['requires_name']}' first ({st['n_left']} question(s) left)")


def log_event(reviewer: str | None, kind: str, detail: dict | None = None) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO events (ts, reviewer, kind, detail) VALUES (?, ?, ?, ?)",
            (_now(), reviewer, kind, json.dumps(detail, ensure_ascii=False) if detail else None),
        )
        conn.commit()


# ---------------------------------------------------------------- reviewers

def list_reviewers(active_only: bool = False) -> list[dict]:
    """Never returns the password hash; `has_password` says whether one is set."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, name, role, active, created_at, password_hash IS NOT NULL AS has_password FROM reviewers"
            + (" WHERE active = 1" if active_only else "") + " ORDER BY name"
        ).fetchall()
    return [dict(r) for r in rows]


def get_reviewer_by_name(name: str) -> dict | None:
    """Includes password_hash — for the login check only, never for a response."""
    with connect() as conn:
        r = conn.execute("SELECT id, name, role, active, password_hash FROM reviewers WHERE name = ?",
                         (" ".join(name.split()),)).fetchone()
    return dict(r) if r else None


def get_reviewer(rid: int) -> dict | None:
    with connect() as conn:
        r = conn.execute("SELECT id, name, role, active, password_hash FROM reviewers WHERE id = ?", (rid,)).fetchone()
    return dict(r) if r else None


def add_reviewer(name: str, role: str = "rater", password_hash: str | None = None) -> dict:
    name = " ".join(name.split())
    if not name:
        raise ValueError("empty name")
    with connect() as conn:
        try:
            conn.execute("INSERT INTO reviewers (name, role, active, password_hash, created_at) VALUES (?, ?, 1, ?, ?)",
                         (name, role, password_hash, _now()))
            conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError(f"reviewer '{name}' already exists")
        r = conn.execute("SELECT id, name, role, active, created_at, password_hash IS NOT NULL AS has_password "
                         "FROM reviewers WHERE name = ?", (name,)).fetchone()
    return dict(r)


def set_password(rid: int, password_hash: str) -> None:
    with connect() as conn:
        if conn.execute("UPDATE reviewers SET password_hash = ? WHERE id = ?", (password_hash, rid)).rowcount == 0:
            raise KeyError(rid)
        conn.commit()


def update_reviewer(rid: int, name: str | None = None, active: bool | None = None, role: str | None = None) -> dict:
    with connect() as conn:
        if name is not None:
            name = " ".join(name.split())
            if not name:
                raise ValueError("name must not be empty")
            try:
                conn.execute("UPDATE reviewers SET name = ? WHERE id = ?", (name, rid))
            except sqlite3.IntegrityError:
                raise ValueError(f"name '{name}' is already taken")
        if active is not None:
            conn.execute("UPDATE reviewers SET active = ? WHERE id = ?", (1 if active else 0, rid))
        if role is not None:
            if role not in ("rater", "admin"):
                raise ValueError("role must be rater or admin")
            conn.execute("UPDATE reviewers SET role = ? WHERE id = ?", (role, rid))
        conn.commit()
        r = conn.execute("SELECT id, name, role, active, created_at, password_hash IS NOT NULL AS has_password "
                         "FROM reviewers WHERE id = ?", (rid,)).fetchone()
    if not r:
        raise KeyError(rid)
    return dict(r)


# ---------------------------------------------------------------- campaigns

def list_sample_files() -> list[dict]:
    out = []
    for p in sorted(SAMPLE_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text())
            n = len(d["questions"]) if isinstance(d, dict) else len(d)
            meta = d.get("meta", {}) if isinstance(d, dict) else {}
        except Exception:  # noqa: BLE001 — a broken file is still worth listing
            n, meta = 0, {"error": "unreadable"}
        out.append({"file": p.name, "n": n, "meta": meta})
    return out


def _campaign_row(conn: sqlite3.Connection, cid: int) -> dict | None:
    r = conn.execute("SELECT * FROM campaigns WHERE id = ?", (cid,)).fetchone()
    if not r:
        return None
    d = dict(r)
    d["n_questions"] = conn.execute("SELECT COUNT(*) FROM questions WHERE campaign_id = ?", (cid,)).fetchone()[0]
    d["n_main"] = conn.execute("SELECT COUNT(*) FROM questions WHERE campaign_id = ? AND batch = 'main'", (cid,)).fetchone()[0]
    d["n_diag"] = d["n_questions"] - d["n_main"]
    return d


def list_campaigns() -> list[dict]:
    with connect() as conn:
        ids = [r[0] for r in conn.execute("SELECT id FROM campaigns ORDER BY id").fetchall()]
        return [_campaign_row(conn, i) for i in ids]


def get_campaign(cid: int) -> dict | None:
    with connect() as conn:
        return _campaign_row(conn, cid)


def set_campaign_status(cid: int, status: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE campaigns SET status = ? WHERE id = ?", (status, cid))
        conn.commit()


def _fetch_chunks_from_kg(chunk_ids: list[str]) -> dict[str, dict]:
    """Read the gold clauses' text from the live KG. Lazy import: the review
    package must import even where the neo4j driver is absent (tests)."""
    from neo4j import GraphDatabase  # noqa: PLC0415

    uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    auth = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))
    out: dict[str, dict] = {}
    driver = GraphDatabase.driver(uri, auth=auth)
    try:
        with driver.session() as s:
            for i in range(0, len(chunk_ids), 200):
                batch = chunk_ids[i:i + 200]
                res = s.run(
                    "MATCH (c:Chunk) WHERE c.chunk_id IN $ids "
                    "RETURN c.chunk_id AS chunk_id, c.spec_id AS spec_id, c.section_id AS section_id, "
                    "c.section_title AS section_title, c.content AS content",
                    ids=batch,
                )
                for r in res:
                    out[r["chunk_id"]] = dict(r)
    finally:
        driver.close()
    return out


def create_campaign(name: str, sample_file: str, raters_per_q: int, seed: int,
                    reviewer_ids: list[int], snapshot: bool = True, all_raters: bool = False,
                    requires_campaign_id: int | None = None) -> dict:
    name = " ".join(name.split())
    if not name:
        raise ValueError("empty campaign name")
    if all_raters:
        raters_per_q = max(1, len(reviewer_ids))
    if raters_per_q < 1:
        raise ValueError("raters_per_q must be >= 1")
    path = SAMPLE_DIR / Path(sample_file).name
    if not path.is_file():
        raise ValueError(f"sample file not found: {sample_file}")
    data = json.loads(path.read_text())
    questions = data["questions"] if isinstance(data, dict) else data
    if not questions:
        raise ValueError("sample file has no questions")

    # Validate the rater list BEFORE writing anything, so a refused assignment
    # cannot leave a half-created campaign behind.
    if reviewer_ids:
        with connect() as conn:
            bad = [rid for rid in set(reviewer_ids)
                   if not conn.execute("SELECT 1 FROM reviewers WHERE id = ? AND active = 1", (rid,)).fetchone()]
        if bad:
            raise ValueError(f"reviewer(s) {sorted(bad)} missing or inactive")
        if not all_raters and len(set(reviewer_ids)) < raters_per_q:
            raise ValueError(f"need at least {raters_per_q} reviewers, got {len(set(reviewer_ids))}")

    chunk_ids = sorted({c for q in questions for c in q["gold_chunk_ids"]})
    chunks = _fetch_chunks_from_kg(chunk_ids) if snapshot else {}
    tables = T.load_tables(chunk_ids)

    with connect() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO campaigns (name, sample_file, seed, raters_per_q, status, created_at, all_raters, requires_campaign_id) "
                "VALUES (?, ?, ?, ?, 'open', ?, ?, ?)",
                (name, path.name, seed, raters_per_q, _now(), 1 if all_raters else 0, requires_campaign_id),
            )
        except sqlite3.IntegrityError:
            raise ValueError(f"campaign '{name}' already exists")
        cid = cur.lastrowid
        for pos, q in enumerate(questions):
            conn.execute(
                "INSERT INTO questions (campaign_id, qid, pos, batch, question, question_class, question_subclass, "
                "question_type, spec_series, gold_chunk_ids, reference_answer, judge_gold) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (cid, q["qid"], pos, q.get("batch", "main"), q["question"], q.get("question_class"),
                 q.get("question_subclass"), q.get("question_type"), q.get("spec_series"),
                 json.dumps(q["gold_chunk_ids"]), q["reference_answer"],
                 json.dumps(q.get("judge_gold") or {})),
            )
        for c in chunk_ids:
            r = chunks.get(c)
            conn.execute(
                "INSERT INTO gold_chunks (campaign_id, chunk_id, spec_id, section_id, section_title, content, n_chars, tables_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (cid, c, r["spec_id"] if r else None, r["section_id"] if r else None,
                 r["section_title"] if r else None, r["content"] if r else None,
                 len(r["content"] or "") if r else None, json.dumps(tables.get(c, []), ensure_ascii=False)),
            )
        conn.commit()
    missing = [c for c in chunk_ids if c not in chunks] if snapshot else chunk_ids
    log_event(None, "campaign_created", {"campaign": cid, "name": name, "n": len(questions), "missing_chunks": missing})
    summary = assign(cid, reviewer_ids) if reviewer_ids else {}
    out = get_campaign(cid)
    out["missing_chunks"] = missing
    out["assignment"] = summary
    return out


# ---------------------------------------------------------------- assignment

def assign(cid: int, reviewer_ids: list[int]) -> dict:
    """(Re)assign every question of the campaign to `raters_per_q` reviewers.

    Pairs that already carry a rating are never touched (a verdict must keep
    its question). Everything else is recomputed: the lowest-loaded reviewers
    take each question, ties broken by a seeded RNG, so with N reviewers and
    k raters per question each reviewer ends up with ~k*Q/N questions and a
    class mix that matches the set. A reviewer's queue order is a seeded
    shuffle, kept stable for pairs that already existed."""
    with connect() as conn:
        camp = conn.execute("SELECT seed, raters_per_q, all_raters, requires_campaign_id FROM campaigns WHERE id = ?",
                            (cid,)).fetchone()
        if not camp:
            raise KeyError(cid)
        reviewer_ids = sorted(set(int(r) for r in reviewer_ids))
        k = len(reviewer_ids) if camp["all_raters"] else camp["raters_per_q"]
        if len(reviewer_ids) < k:
            raise ValueError(f"need at least {k} reviewers, got {len(reviewer_ids)}")
        for rid in reviewer_ids:
            if not conn.execute("SELECT 1 FROM reviewers WHERE id = ? AND active = 1", (rid,)).fetchone():
                raise ValueError(f"reviewer {rid} missing or inactive")
        qrows = conn.execute("SELECT qid, question_class FROM questions WHERE campaign_id = ? AND in_scope = 1 ORDER BY pos",
                             (cid,)).fetchall()
        qids = [r[0] for r in qrows]
        cls_of = {r[0]: r[1] or "" for r in qrows}
        rated = {(r[0], r[1]) for r in conn.execute(
            "SELECT qid, reviewer_id FROM ratings WHERE campaign_id = ?", (cid,)).fetchall()}
        old = {(r[0], r[1]): r[2] for r in conn.execute(
            "SELECT qid, reviewer_id, order_idx FROM assignments WHERE campaign_id = ?", (cid,)).fetchall()}

        rnd = random.Random(f"{camp['seed']}:{cid}")
        load = {rid: 0 for rid in reviewer_ids}
        # Balanced PER CLASS as well as in total: a procedure question carries
        # ~8x the gold text of a requirement one, so equal counts alone can
        # hand one rater hours more reading than another.
        load_cls: dict[tuple[int, str], int] = {}
        # Reviewers no longer in the list keep only the pairs they already rated.
        keep: set[tuple[str, int]] = set()
        for (q, rid) in rated:
            keep.add((q, rid))
            if rid in load:
                load[rid] += 1
                load_cls[(rid, cls_of.get(q, ""))] = load_cls.get((rid, cls_of.get(q, "")), 0) + 1
        order = qids[:]
        rnd.shuffle(order)
        for q in order:
            have = {rid for (qq, rid) in keep if qq == q}
            need = k - len(have)
            if need <= 0:
                continue
            cands = [rid for rid in reviewer_ids if rid not in have]
            rnd.shuffle(cands)
            c = cls_of.get(q, "")
            cands.sort(key=lambda rid: (load_cls.get((rid, c), 0), load[rid]))
            for rid in cands[:need]:
                keep.add((q, rid))
                load[rid] += 1
                load_cls[(rid, c)] = load_cls.get((rid, c), 0) + 1

        conn.execute("DELETE FROM assignments WHERE campaign_id = ?", (cid,))
        per_reviewer: dict[int, list[str]] = {}
        for (q, rid) in keep:
            per_reviewer.setdefault(rid, []).append(q)
        for rid, qs in per_reviewer.items():
            kept = sorted((q for q in qs if (q, rid) in old), key=lambda q: old[(q, rid)])
            # sorted() first: `qs` comes from set iteration, whose order changes between
            # processes, so without it the seeded shuffle would not be reproducible
            new = sorted(q for q in qs if (q, rid) not in old)
            r2 = random.Random(f"{camp['seed']}:{cid}:{rid}")
            r2.shuffle(new)
            for i, q in enumerate(kept + new):
                conn.execute(
                    "INSERT INTO assignments (campaign_id, qid, reviewer_id, order_idx) VALUES (?, ?, ?, ?)",
                    (cid, q, rid, i),
                )
        conn.commit()
    summary = {str(rid): len(qs) for rid, qs in sorted(per_reviewer.items())}
    log_event(None, "assigned", {"campaign": cid, "per_reviewer": summary})
    if camp["requires_campaign_id"]:
        ensure_in_prerequisite(cid)
    return summary


def auto_enroll(rid: int) -> list[int]:
    """Put a (new) rater into every OPEN campaign where every rater rates every
    question — the pilot and the main 'all raters' campaign — so a new account
    gets the full set without the admin assigning it by hand."""
    with connect() as conn:
        cids = [r[0] for r in conn.execute(
            "SELECT id FROM campaigns WHERE status = 'open' AND all_raters = 1 ORDER BY id")]
    done = []
    for cid in cids:
        with connect() as conn:
            current = {r[0] for r in conn.execute(
                "SELECT DISTINCT a.reviewer_id FROM assignments a JOIN reviewers v ON v.id = a.reviewer_id "
                "WHERE a.campaign_id = ? AND v.active = 1", (cid,))}
        if rid not in current:
            assign(cid, sorted(current | {rid}))
            done.append(cid)
    return done


def ensure_in_prerequisite(cid: int) -> None:
    """Everyone assigned in `cid` must also be in its prerequisite (pilot),
    otherwise they would be locked out with nothing to unlock them."""
    with connect() as conn:
        row = conn.execute("SELECT requires_campaign_id FROM campaigns WHERE id = ?", (cid,)).fetchone()
        req = row[0] if row else None
        if not req:
            return
        active = "SELECT DISTINCT a.reviewer_id FROM assignments a JOIN reviewers r ON r.id = a.reviewer_id " \
                 "WHERE a.campaign_id = ? AND r.active = 1"
        here = {r[0] for r in conn.execute(active, (cid,))}
        there = {r[0] for r in conn.execute(active, (req,))}
    if here - there:
        assign(req, sorted(here | there))


def set_scope(cid: int, qids: list[str]) -> dict:
    """Restrict the campaign to `qids`; others keep their ratings but leave the queues."""
    keep = set(qids)
    with connect() as conn:
        have = {r[0] for r in conn.execute("SELECT qid FROM questions WHERE campaign_id = ?", (cid,))}
        missing = keep - have
        if missing:
            raise ValueError(f"{len(missing)} qid(s) not in this campaign, e.g. {sorted(missing)[:3]}")
        conn.execute("UPDATE questions SET in_scope = CASE WHEN qid IN (%s) THEN 1 ELSE 0 END WHERE campaign_id = ?"
                     % ",".join("?" * len(keep)), (*sorted(keep), cid))
        # drop not-yet-rated assignments of out-of-scope questions
        conn.execute("DELETE FROM assignments WHERE campaign_id = ? AND qid IN (SELECT qid FROM questions WHERE campaign_id = ? AND in_scope = 0) "
                     "AND NOT EXISTS (SELECT 1 FROM ratings r WHERE r.campaign_id = assignments.campaign_id AND r.qid = assignments.qid "
                     "AND r.reviewer_id = assignments.reviewer_id)", (cid, cid))
        conn.commit()
    log_event(None, "scope_set", {"campaign": cid, "n_in_scope": len(keep)})
    return get_campaign(cid)


def set_all_raters(cid: int, on: bool) -> None:
    with connect() as conn:
        conn.execute("UPDATE campaigns SET all_raters = ? WHERE id = ?", (1 if on else 0, cid))
        conn.commit()


def set_requirement(cid: int, requires_campaign_id: int | None) -> dict:
    if requires_campaign_id == cid:
        raise ValueError("a campaign cannot require itself")
    with connect() as conn:
        if requires_campaign_id and not conn.execute("SELECT 1 FROM campaigns WHERE id = ?", (requires_campaign_id,)).fetchone():
            raise ValueError("required campaign does not exist")
        conn.execute("UPDATE campaigns SET requires_campaign_id = ? WHERE id = ?", (requires_campaign_id, cid))
        conn.commit()
    ensure_in_prerequisite(cid)
    return get_campaign(cid)


def delete_campaign(cid: int) -> None:
    """Only an untouched campaign can be deleted — ratings are paper data."""
    with connect() as conn:
        if conn.execute("SELECT COUNT(*) FROM ratings WHERE campaign_id = ?", (cid,)).fetchone()[0]:
            raise ValueError("campaign has ratings; close it instead of deleting")
        if conn.execute("SELECT 1 FROM campaigns WHERE requires_campaign_id = ?", (cid,)).fetchone():
            raise ValueError("another campaign requires this one; remove that requirement first")
        conn.execute("DELETE FROM assignments WHERE campaign_id = ?", (cid,))
        conn.execute("DELETE FROM adjudications WHERE campaign_id = ?", (cid,))
        conn.execute("DELETE FROM campaigns WHERE id = ?", (cid,))
        conn.commit()
    log_event(None, "campaign_deleted", {"campaign": cid})


# ---------------------------------------------------------------- rater side

def campaigns_for_reviewer(rid: int) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT c.id, c.name, c.status, c.all_raters, c.requires_campaign_id, COUNT(a.qid) AS n_assigned, "
            "SUM(CASE WHEN r.status = 'done' THEN 1 ELSE 0 END) AS n_done "
            "FROM campaigns c JOIN assignments a ON a.campaign_id = c.id AND a.reviewer_id = ? "
            "JOIN questions qs ON qs.campaign_id = a.campaign_id AND qs.qid = a.qid AND qs.in_scope = 1 "
            "LEFT JOIN ratings r ON r.campaign_id = a.campaign_id AND r.qid = a.qid AND r.reviewer_id = a.reviewer_id "
            # Closed campaigns are kept for the record but no longer shown to raters.
            "WHERE c.status = 'open' GROUP BY c.id ORDER BY c.id", (rid,)
        ).fetchall()
        required = {r[0] for r in conn.execute("SELECT requires_campaign_id FROM campaigns WHERE requires_campaign_id IS NOT NULL")}
        out = []
        for r in rows:
            d = dict(r)
            d["n_done"] = d["n_done"] or 0
            d["lock"] = _lock_state(conn, d["id"], rid)
            # A pilot is a campaign another one requires — not merely "everyone rates
            # everything", which the main campaign now is too.
            d["is_pilot"] = d["id"] in required
            out.append(d)
    # Pilots first, so the default campaign a rater lands on is the one they must do.
    out.sort(key=lambda d: (not d["is_pilot"], d["id"]))
    return out


def queue_for(cid: int, rid: int) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT a.qid, a.order_idx, r.status, r.revealed_at, r.updated_at "
            "FROM assignments a " "JOIN questions qs ON qs.campaign_id = a.campaign_id AND qs.qid = a.qid AND qs.in_scope = 1 " "LEFT JOIN ratings r "
            "ON r.campaign_id = a.campaign_id AND r.qid = a.qid AND r.reviewer_id = a.reviewer_id "
            "WHERE a.campaign_id = ? AND a.reviewer_id = ? ORDER BY a.order_idx", (cid, rid)
        ).fetchall()
    out = []
    for i, r in enumerate(rows):
        out.append({"qid": r["qid"], "idx": i + 1, "status": r["status"] or "todo",
                    "revealed": bool(r["revealed_at"]), "updated_at": r["updated_at"]})
    return out


def _assigned(conn: sqlite3.Connection, cid: int, qid: str, rid: int) -> bool:
    return conn.execute(
        "SELECT 1 FROM assignments WHERE campaign_id = ? AND qid = ? AND reviewer_id = ?", (cid, qid, rid)
    ).fetchone() is not None


def _rating_dict(r: sqlite3.Row | None) -> dict | None:
    if r is None:
        return None
    d = dict(r)
    d["irrelevant_chunks"] = json.loads(d.get("irrelevant_chunks") or "[]")
    return d


def question_view(cid: int, qid: str, rid: int) -> dict:
    """What a rater sees. The reference answer is included ONLY once this rater
    has revealed it (phase 2) — the blind phase is enforced server-side."""
    with connect() as conn:
        if not _assigned(conn, cid, qid, rid):
            raise PermissionError("not assigned")
        _check_unlocked(conn, cid, rid)
        q = conn.execute("SELECT * FROM questions WHERE campaign_id = ? AND qid = ?", (cid, qid)).fetchone()
        if not q:
            raise KeyError(qid)
        gold_ids = json.loads(q["gold_chunk_ids"])
        chunks = {r["chunk_id"]: dict(r) for r in conn.execute(
            "SELECT chunk_id, spec_id, section_id, section_title, content, n_chars, tables_json FROM gold_chunks "
            "WHERE campaign_id = ? AND chunk_id IN (%s)" % ",".join("?" * len(gold_ids)), (cid, *gold_ids)
        ).fetchall()}
        rating = _rating_dict(conn.execute(
            "SELECT * FROM ratings WHERE campaign_id = ? AND qid = ? AND reviewer_id = ?", (cid, qid, rid)).fetchone())
        idx = conn.execute(
            "SELECT order_idx FROM assignments WHERE campaign_id = ? AND qid = ? AND reviewer_id = ?", (cid, qid, rid)
        ).fetchone()[0]
        total = conn.execute(
            "SELECT COUNT(*) FROM assignments a " "JOIN questions qs ON qs.campaign_id = a.campaign_id AND qs.qid = a.qid AND qs.in_scope = 1 " "WHERE a.campaign_id = ? AND a.reviewer_id = ?",
            (cid, rid)).fetchone()[0]
        nxt = conn.execute(
            "SELECT a.qid FROM assignments a " "JOIN questions qs ON qs.campaign_id = a.campaign_id AND qs.qid = a.qid AND qs.in_scope = 1 " "LEFT JOIN ratings r ON r.campaign_id = a.campaign_id AND r.qid = a.qid "
            "AND r.reviewer_id = a.reviewer_id WHERE a.campaign_id = ? AND a.reviewer_id = ? "
            "AND (r.status IS NULL OR r.status != 'done') AND a.order_idx > ? ORDER BY a.order_idx LIMIT 1",
            (cid, rid, idx)).fetchone()
    revealed = bool(rating and rating.get("revealed_at"))
    gold = []
    for c in gold_ids:
        r = chunks.get(c)
        gold.append({
            "chunk_id": c,
            "spec_id": r["spec_id"] if r else None,
            "section_id": r["section_id"] if r else None,
            "section_title": r["section_title"] if r else None,
            "content": r["content"] if r else None,
            "n_chars": r["n_chars"] if r else None,
            "missing": not r or r["content"] is None,
            "segments": T.segments(r["content"], json.loads(r["tables_json"] or "[]")) if r and r["content"] else [],
        })
    return {
        "qid": qid,
        "idx": idx + 1,
        "total": total,
        "question": q["question"],
        "gold": gold,
        "rating": rating,
        "revealed": revealed,
        "reference_answer": q["reference_answer"] if revealed else None,
        "next_qid": nxt[0] if nxt else None,
    }


def _phase1_complete(r: dict | None) -> bool:
    return bool(r and r.get("c1") in C1 and r.get("c2") in C2 and (r.get("own_answer") or "").strip()
                and (r.get("c1") == "pass" or (r.get("c1_note") or "").strip()))


def reveal(cid: int, qid: str, rid: int, reviewer_name: str) -> dict:
    with connect() as conn:
        if not _assigned(conn, cid, qid, rid):
            raise PermissionError("not assigned")
        _check_unlocked(conn, cid, rid)
        rating = _rating_dict(conn.execute(
            "SELECT * FROM ratings WHERE campaign_id = ? AND qid = ? AND reviewer_id = ?", (cid, qid, rid)).fetchone())
        if not _phase1_complete(rating):
            raise ValueError("Step 1 incomplete: C1, C2 (a reason if C1 is not Pass) and your own answer are required")
        if not rating.get("revealed_at"):
            conn.execute("UPDATE ratings SET revealed_at = ?, updated_at = ? WHERE campaign_id = ? AND qid = ? AND reviewer_id = ?",
                         (_now(), _now(), cid, qid, rid))
            conn.commit()
            # Freeze what the rater said BEFORE seeing the answer; later edits to
            # phase-1 fields are allowed but become visible against this record.
            log_event(reviewer_name, "reveal", {"campaign": cid, "qid": qid,
                                                "c1": rating["c1"], "c2": rating["c2"],
                                                "irrelevant_chunks": rating["irrelevant_chunks"],
                                                "own_answer": rating["own_answer"]})
    return question_view(cid, qid, rid)


def upsert_rating(cid: int, qid: str, rid: int, p: dict, status: str) -> dict:
    if status not in ("draft", "done"):
        raise ValueError("bad status")
    for k in ("c1", "c2", "c3", "verdict"):
        v = p.get(k)
        if v is not None and v not in LABELS[k]:
            raise ValueError(f"bad {k}: {v}")
    with connect() as conn:
        if not _assigned(conn, cid, qid, rid):
            raise PermissionError("not assigned")
        _check_unlocked(conn, cid, rid)
        if conn.execute("SELECT status FROM campaigns WHERE id = ?", (cid,)).fetchone()[0] != "open":
            raise ValueError("This campaign is closed")
        cur = _rating_dict(conn.execute(
            "SELECT * FROM ratings WHERE campaign_id = ? AND qid = ? AND reviewer_id = ?", (cid, qid, rid)).fetchone())
        merged = dict(cur or {})
        for k in ("c1", "c1_note", "c2", "c2_note", "own_answer", "c3", "verdict", "comment", "fix_text",
                  "t_phase1_s", "t_phase2_s"):
            if k in p:
                merged[k] = p[k]
        if "irrelevant_chunks" in p:
            merged["irrelevant_chunks"] = list(p["irrelevant_chunks"] or [])
        if status == "done":
            if not _phase1_complete(merged):
                raise ValueError("Step 1 incomplete: C1, C2 (a reason if C1 is not Pass) and your own answer are required")
            if not merged.get("revealed_at"):
                raise ValueError("Reference answer not revealed yet (Step 2)")
            if merged.get("c3") not in C3 or merged.get("verdict") not in VERDICT:
                raise ValueError("Step 2 incomplete: C3 and a verdict are required")
            if merged["verdict"] != "keep" and not (merged.get("comment") or "").strip():
                raise ValueError("A comment is required when the verdict is Fix or Discard")
        now = _now()
        if cur is None:
            conn.execute(
                "INSERT INTO ratings (campaign_id, qid, reviewer_id, status, c1, c1_note, c2, c2_note, irrelevant_chunks, "
                "own_answer, c3, verdict, comment, fix_text, t_phase1_s, t_phase2_s, revealed_at, revision, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, ?, ?)",
                (cid, qid, rid, status, merged.get("c1"), merged.get("c1_note"), merged.get("c2"), merged.get("c2_note"),
                 json.dumps(merged.get("irrelevant_chunks") or []), merged.get("own_answer"), merged.get("c3"),
                 merged.get("verdict"), merged.get("comment"), merged.get("fix_text"), merged.get("t_phase1_s"),
                 merged.get("t_phase2_s"), now, now),
            )
        else:
            conn.execute(
                "UPDATE ratings SET status = ?, c1 = ?, c1_note = ?, c2 = ?, c2_note = ?, irrelevant_chunks = ?, own_answer = ?, "
                "c3 = ?, verdict = ?, comment = ?, fix_text = ?, t_phase1_s = ?, t_phase2_s = ?, "
                "revision = revision + ?, updated_at = ? WHERE campaign_id = ? AND qid = ? AND reviewer_id = ?",
                (status, merged.get("c1"), merged.get("c1_note"), merged.get("c2"), merged.get("c2_note"),
                 json.dumps(merged.get("irrelevant_chunks") or []), merged.get("own_answer"), merged.get("c3"),
                 merged.get("verdict"), merged.get("comment"), merged.get("fix_text"), merged.get("t_phase1_s"),
                 merged.get("t_phase2_s"), 1 if cur.get("status") == "done" else 0, now, cid, qid, rid),
            )
        conn.commit()
        return _rating_dict(conn.execute(
            "SELECT * FROM ratings WHERE campaign_id = ? AND qid = ? AND reviewer_id = ?", (cid, qid, rid)).fetchone())


# ---------------------------------------------------------------- admin side

def progress(cid: int) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT rv.id AS reviewer_id, rv.name, COUNT(a.qid) AS n_assigned, "
            "SUM(CASE WHEN r.status = 'done' THEN 1 ELSE 0 END) AS n_done, "
            "SUM(CASE WHEN r.status = 'draft' THEN 1 ELSE 0 END) AS n_draft, "
            "AVG(CASE WHEN r.status = 'done' THEN COALESCE(r.t_phase1_s,0) + COALESCE(r.t_phase2_s,0) END) AS avg_seconds, "
            "MAX(r.updated_at) AS last_activity "
            "FROM reviewers rv JOIN assignments a ON a.reviewer_id = rv.id AND a.campaign_id = ? "
            "JOIN questions qs ON qs.campaign_id = a.campaign_id AND qs.qid = a.qid AND qs.in_scope = 1 "
            "LEFT JOIN ratings r ON r.campaign_id = a.campaign_id AND r.qid = a.qid AND r.reviewer_id = a.reviewer_id "
            "GROUP BY rv.id ORDER BY rv.name", (cid,)
        ).fetchall()
    return [dict(r) for r in rows]


def _load_done(conn: sqlite3.Connection, cid: int) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    qs = {r["qid"]: dict(r) for r in conn.execute("SELECT * FROM questions WHERE campaign_id = ?", (cid,)).fetchall()}
    for q in qs.values():
        q["gold_chunk_ids"] = json.loads(q["gold_chunk_ids"])
        q["judge_gold"] = json.loads(q["judge_gold"] or "{}")
    ratings: dict[str, list[dict]] = {}
    for r in conn.execute("SELECT * FROM ratings WHERE campaign_id = ? AND status = 'done' ORDER BY reviewer_id", (cid,)).fetchall():
        ratings.setdefault(r["qid"], []).append(_rating_dict(r))
    return qs, ratings


def _judge_q_label(judge: dict[str, str]) -> str | None:
    if not judge:
        return None
    return min(judge.values(), key=lambda l: _JUDGE_RANK.get(l, 1))


def stats(cid: int) -> dict:
    with connect() as conn:
        qs, ratings = _load_done(conn, cid)
        adj = {r["qid"]: dict(r) for r in conn.execute("SELECT * FROM adjudications WHERE campaign_id = ?", (cid,)).fetchall()}
    classes = sorted({q["question_class"] for q in qs.values() if q["question_class"]})

    def block(sel: list[str]) -> dict:
        out: dict = {"n_questions": len(sel),
                     "n_rated_2plus": sum(1 for q in sel if len(ratings.get(q, [])) >= 2),
                     "n_rated_1": sum(1 for q in sel if len(ratings.get(q, [])) == 1),
                     "criteria": {}}
        for crit, labels in LABELS.items():
            items = [[r[crit] for r in ratings.get(q, []) if r.get(crit)] for q in sel]
            flat = [l for it in items for l in it]
            per_q = [it for it in items if len(it) >= 2]
            out["criteria"][crit] = {
                "labels": list(labels),
                "dist": {l: flat.count(l) for l in labels},
                "pass_rating": S.rate(flat.count(PASS[crit]), len(flat)),
                "pass_all_raters": S.rate(sum(1 for it in per_q if all(l == PASS[crit] for l in it)), len(per_q)),
                "pass_any_rater": S.rate(sum(1 for it in per_q if any(l == PASS[crit] for l in it)), len(per_q)),
                "agreement": S.agreement_block(items, len(labels)),
                "pairwise": S.pairwise_cohen({q: {r["reviewer_id"]: r[crit] for r in ratings.get(q, []) if r.get(crit)} for q in sel}),
            }
        all3 = [[r["c1"] == "pass" and r["c2"] == "full" and r["c3"] == "correct" for r in ratings.get(q, [])] for q in sel]
        flat3 = [x for it in all3 for x in it]
        per3 = [it for it in all3 if len(it) >= 2]
        out["all_three"] = {
            "pass_rating": S.rate(sum(flat3), len(flat3)),
            "pass_all_raters": S.rate(sum(1 for it in per3 if all(it)), len(per3)),
            "pass_any_rater": S.rate(sum(1 for it in per3 if any(it)), len(per3)),
            "agreement": S.agreement_block([["y" if x else "n" for x in it] for it in all3], 2),
        }
        out["adjudication"] = {d: sum(1 for q in sel if adj.get(q, {}).get("decision") == d) for d in DECISIONS}
        out["adjudication"]["pending"] = sum(1 for q in sel if q not in adj and len(ratings.get(q, [])) >= 2
                                            and _needs_adjudication(ratings[q]))
        return out

    def judge_block(sel: list[str]) -> dict:
        chunk_conf: dict[str, dict[str, int]] = {l: {"relevant": 0, "irrelevant": 0} for l in _JUDGE_RANK}
        q_conf: dict[str, dict[str, int]] = {l: {c: 0 for c in C2} for l in _JUDGE_RANK}
        pairs_j, pairs_h = [], []
        for q in sel:
            judge = qs[q]["judge_gold"]
            jq = _judge_q_label(judge)
            for r in ratings.get(q, []):
                if jq:
                    q_conf[jq][r["c2"]] += 1
                irr = set(r["irrelevant_chunks"])
                for c, jl in judge.items():
                    h = "irrelevant" if c in irr else "relevant"
                    chunk_conf[jl][h] += 1
                    pairs_j.append("irrelevant" if jl == "NOT_RELEVANT" else "relevant")
                    pairs_h.append(h)
        return {
            "chunk_level": {"confusion": chunk_conf, "n": len(pairs_j),
                            "raw": round(sum(1 for a, b in zip(pairs_j, pairs_h) if a == b) / len(pairs_j), 4) if pairs_j else None,
                            "cohen_kappa_binary": S.cohen_kappa(pairs_j, pairs_h)},
            "question_level": {"confusion": q_conf},
        }

    out: dict = {"campaign": cid, "classes": classes, "batches": {}}
    for batch in ("main", "diag"):
        # out-of-scope main questions (dropped when the set became 295/rater) are not
        # part of the main measurement; the diag batch is reported whatever its scope
        sel_all = [q for q, d in qs.items() if d["batch"] == batch and (batch == "diag" or d.get("in_scope", 1))]
        if not sel_all:
            continue
        b: dict = {"all": block(sel_all), "judge": judge_block(sel_all), "per_class": {}}
        for cls in classes:
            sel = [q for q in sel_all if qs[q]["question_class"] == cls]
            if sel:
                b["per_class"][cls] = block(sel)
                b["per_class"][cls]["judge"] = judge_block(sel)
        out["batches"][batch] = b
    out["progress"] = progress(cid)
    return out


def _needs_adjudication(rs: list[dict]) -> bool:
    if any(r["verdict"] != "keep" for r in rs):
        return True
    for crit in ("c1", "c2", "c3", "verdict"):
        if len({r[crit] for r in rs}) > 1:
            return True
    return False


def disagreements(cid: int) -> list[dict]:
    with connect() as conn:
        qs, ratings = _load_done(conn, cid)
        adj = {r["qid"]: dict(r) for r in conn.execute("SELECT * FROM adjudications WHERE campaign_id = ?", (cid,)).fetchall()}
        names = {r["id"]: r["name"] for r in conn.execute("SELECT id, name FROM reviewers").fetchall()}
    out = []
    for q, rs in ratings.items():
        if len(rs) < 2 or not _needs_adjudication(rs):
            continue
        out.append({
            "qid": q, "batch": qs[q]["batch"], "question_class": qs[q]["question_class"],
            "question": qs[q]["question"],
            "ratings": [{"reviewer": names.get(r["reviewer_id"], r["reviewer_id"]), **{k: r[k] for k in ("c1", "c2", "c3", "verdict", "comment")}} for r in rs],
            "adjudication": adj.get(q),
        })
    out.sort(key=lambda d: (d["adjudication"] is not None, d["qid"]))
    return out


def adjudicate(cid: int, qid: str, decision: str, note: str | None, by: str | None) -> dict:
    if decision not in DECISIONS:
        raise ValueError("bad decision")
    with connect() as conn:
        if not conn.execute("SELECT 1 FROM questions WHERE campaign_id = ? AND qid = ?", (cid, qid)).fetchone():
            raise KeyError(qid)
        conn.execute(
            "INSERT INTO adjudications (campaign_id, qid, decision, note, decided_by, created_at) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(campaign_id, qid) DO UPDATE SET decision = excluded.decision, note = excluded.note, "
            "decided_by = excluded.decided_by, created_at = excluded.created_at",
            (cid, qid, decision, note, by, _now()),
        )
        conn.commit()
        return dict(conn.execute("SELECT * FROM adjudications WHERE campaign_id = ? AND qid = ?", (cid, qid)).fetchone())


def admin_question(cid: int, qid: str) -> dict:
    with connect() as conn:
        q = conn.execute("SELECT * FROM questions WHERE campaign_id = ? AND qid = ?", (cid, qid)).fetchone()
        if not q:
            raise KeyError(qid)
        q = dict(q)
        q["gold_chunk_ids"] = json.loads(q["gold_chunk_ids"])
        q["judge_gold"] = json.loads(q["judge_gold"] or "{}")
        ids = q["gold_chunk_ids"]
        chunks = [dict(r) for r in conn.execute(
            "SELECT chunk_id, spec_id, section_id, section_title, content, n_chars, tables_json FROM gold_chunks "
            "WHERE campaign_id = ? AND chunk_id IN (%s)" % ",".join("?" * len(ids)), (cid, *ids)).fetchall()]
        for ch in chunks:
            ch["segments"] = T.segments(ch["content"], json.loads(ch.pop("tables_json") or "[]")) if ch["content"] else []
        names = {r["id"]: r["name"] for r in conn.execute("SELECT id, name FROM reviewers").fetchall()}
        ratings = [_rating_dict(r) for r in conn.execute(
            "SELECT * FROM ratings WHERE campaign_id = ? AND qid = ? ORDER BY reviewer_id", (cid, qid)).fetchall()]
        for r in ratings:
            r["reviewer"] = names.get(r["reviewer_id"])
        adj = conn.execute("SELECT * FROM adjudications WHERE campaign_id = ? AND qid = ?", (cid, qid)).fetchone()
        assigned = [names.get(r[0]) for r in conn.execute(
            "SELECT reviewer_id FROM assignments WHERE campaign_id = ? AND qid = ?", (cid, qid)).fetchall()]
    order = {c: i for i, c in enumerate(ids)}
    chunks.sort(key=lambda c: order.get(c["chunk_id"], 99))
    return {"question": q, "gold": chunks, "ratings": ratings, "adjudication": dict(adj) if adj else None,
            "assigned": assigned}


def export_rows(cid: int) -> dict:
    with connect() as conn:
        camp = _campaign_row(conn, cid)
        qs, _ = _load_done(conn, cid)
        names = {r["id"]: r["name"] for r in conn.execute("SELECT id, name FROM reviewers").fetchall()}
        ratings = [_rating_dict(r) for r in conn.execute(
            "SELECT * FROM ratings WHERE campaign_id = ? ORDER BY qid, reviewer_id", (cid,)).fetchall()]
        adj = {r["qid"]: dict(r) for r in conn.execute("SELECT * FROM adjudications WHERE campaign_id = ?", (cid,)).fetchall()}
        events = [dict(r) for r in conn.execute("SELECT * FROM events WHERE kind = 'reveal' ORDER BY id").fetchall()]
    for r in ratings:
        r["reviewer"] = names.get(r["reviewer_id"])
    return {"campaign": camp, "questions": list(qs.values()), "ratings": ratings,
            "adjudications": list(adj.values()), "reveal_events": events, "stats": stats(cid)}


def export_csv(cid: int) -> str:
    import csv  # noqa: PLC0415
    import io  # noqa: PLC0415

    data = export_rows(cid)
    qs = {q["qid"]: q for q in data["questions"]}
    adj = {a["qid"]: a for a in data["adjudications"]}
    cols = ["campaign", "batch", "qid", "question_class", "question_subclass", "question_type", "reviewer", "status",
            "c1", "c1_note", "c2", "c2_note", "irrelevant_chunks", "own_answer", "c3", "verdict", "comment", "fix_text",
            "t_phase1_s", "t_phase2_s", "revealed_at", "revision", "updated_at", "adj_decision", "adj_note"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    for r in data["ratings"]:
        q = qs[r["qid"]]
        a = adj.get(r["qid"], {})
        w.writerow({
            "campaign": data["campaign"]["name"], "batch": q["batch"], "qid": r["qid"],
            "question_class": q["question_class"], "question_subclass": q["question_subclass"],
            "question_type": q["question_type"], "reviewer": r["reviewer"], "status": r["status"],
            "c1": r["c1"], "c1_note": r["c1_note"], "c2": r["c2"], "c2_note": r["c2_note"],
            "irrelevant_chunks": ";".join(r["irrelevant_chunks"]), "own_answer": r["own_answer"],
            "c3": r["c3"], "verdict": r["verdict"], "comment": r["comment"], "fix_text": r["fix_text"],
            "t_phase1_s": r["t_phase1_s"], "t_phase2_s": r["t_phase2_s"], "revealed_at": r["revealed_at"],
            "revision": r["revision"], "updated_at": r["updated_at"],
            "adj_decision": a.get("decision"), "adj_note": a.get("note"),
        })
    return buf.getvalue()
