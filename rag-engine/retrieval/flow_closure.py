"""
Procedure-closure retrieval: gather the whole evidence set of a call flow the question names.

A question about a complete flow ("which entities take part in the 5G AKA procedure of TS 33.501,
including the initiation and the result confirmation ...") needs 3-8 clauses: the flow's own clause
tree, the clauses it cites, and the definition of each service operation it invokes. Text retrieval
finds the main clause and little else (flow-aggregation pilot, 17/10: BM25+Dense R@20 0.53, 6/33
questions complete), and none of the Cypher patterns re-anchors on a procedure (KG R@20 0.09).

    roots    the heading whose title holds every word of the procedure phrase right before
             "of TS xx.yyy" (searched in that spec), the biggest such subtree winning over the
             restatements in interworking / I-SMF chapters; plus every heading of a cited spec that
             a question n-gram names exactly (the question's own scope). Headings come from
             Chunk.heading_ids/heading_path, which recover the body-less headings the chunker drops
             (kg_builder/enrich_heading_path.py) — without them the procedure name is absent from
             the KG for most flows (root found 10/33 -> 30/33 with them).
    closure  each root's clause subtree (chunk_id prefix, parts pN) + REFERENCES_CHUNK targets of the
             main subtree (1 hop, any spec) + the defining clause of every ServiceOperation the
             subtrees describe (DESCRIBES_OPERATION, title naming the operation).
Gold coverage of the closure on the pilot: 128/160 clauses, mean pool 51 chunks.

Each chunk carries `via_flow_closure` (rerank floor bypass, like the reference hop) and an upstream
`score` by role, so the blend orders the main tree before operation definitions before citations.
Read-only; never raises.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading

_STOP = set("a an the of in on for and or to by with from as at its it is are be this that which what "
            "procedure procedures clause ts tr each does do how".split())
# connective words that titles and questions swap freely ("using N26 interface" / "over N26")
_MATCH_STOP = {"using", "via", "over", "interface"}
_PROC_RE = re.compile(r"\b(?:procedure\s+)?(?:of|in)\s+TS\s*(\d{2})\.(\d{3}(?:-\d+)?)")
_SPEC_RE = re.compile(r"\bT[SR]\s*(\d{2})\.(\d{3}(?:-\d+)?)")
_PART_RE = re.compile(r"(p\d+|~\d+)$")
_WORD_RE = re.compile(r"[A-Za-z0-9'/\-]+")
MAX_POOL = 60
ROLE_SCORE = {"main": 1.0, "op_def": 0.8, "named": 0.7, "ref": 0.6,
              "invoked": 0.65, "stage3": 0.6, "msg_def": 0.55,
              "named_in_text": 0.65, "cited_spec": 0.55}
FLOW_NAME_LINKS = os.getenv("FLOW_NAME_LINKS", "1") not in ("0", "false", "")
_GENERIC = {"general", "description", "overview", "introduction", "procedures", "procedure", "requirements",
            "test requirements", "initial conditions", "test procedure", "test purpose", "definitions",
            "abbreviations", "scope", "references", "void"}
_ENTITY_HEADS = {"function", "functions", "server", "servers", "client", "clients", "entity", "entities", "ue", "node"}
_RAW_RE = re.compile(r"[a-z0-9']+")


def _raw(text: str) -> str:
    return " ".join(_RAW_RE.findall((text or "").lower().replace("-", " ").replace("\u00a0", " ")))
# "heuristic": closure computed here from headings + edges (v2); "graph": read it off the Procedure
# layer the KG builder precomputes (kg_builder/procedures.py). Process-wide, set at startup.
FLOW_CLOSURE_SOURCE = os.getenv("FLOW_CLOSURE_SOURCE", "heuristic").lower()

_cache: dict[str, list] = {}
_titles: dict[str, str] = {}
_parents: dict[str, str] = {}
_lock = threading.Lock()
# "llm": the generator picks the root from a shortlist of the cited spec's heading tree (v4);
# "lexical": _main_root's title-word cover (v3). Process-wide, set at startup.
FLOW_ROOT = os.getenv("FLOW_ROOT", "llm").lower()
ROOT_SHORTLIST = 40


def _toks(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9']+", (s or "").lower().replace("-", " ")) if t not in _STOP}


def _headings(session, spec: str) -> list[tuple[str, set[str], int]]:
    """(heading id, title tokens, subtree text size) for every heading of one spec, cached."""
    with _lock:
        if spec in _cache:
            return _cache[spec]
    rows = session.run(
        "MATCH (c:Chunk {spec_id: $sp}) RETURN c.chunk_id AS i, c.section_title AS t, size(c.content) AS n, "
        "c.heading_ids AS hi, c.heading_path AS hp", sp=spec).data()
    title, size, parent = {}, {}, {}
    for r in rows:
        own = _PART_RE.sub("", r["i"])
        if own == r["i"]:
            title.setdefault(own, r["t"])
        size[own] = size.get(own, 0) + (r["n"] or 0)
        chain = [f"{spec}_{sid}" for sid in (r["hi"] or [])] + [own]
        for a, b in zip(chain, chain[1:]):
            if a != b:
                parent.setdefault(b, a)
        for sid, t in zip(r["hi"] or [], r["hp"] or []):
            k = f"{spec}_{sid}"
            title.setdefault(k, t)
            size[k] = size.get(k, 0) + (r["n"] or 0)
    out = [(i, _toks(t), size.get(i, 0)) for i, t in title.items() if _toks(t)]
    with _lock:
        _cache[spec] = out
        _titles.update(title)
        _parents.update(parent)
    return out


def _stem(t: str) -> str:
    for suf in ("ations", "ation", "ings", "ing", "ments", "ment", "ed", "es", "s", "e"):
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            return t[: -len(suf)]
    return t


def _qtoks(text: str) -> list[str]:
    """Ordered content tokens. An all-caps word is kept even when it spells a stopword ("AN"
    Release is not the article), and "to"/"from" are kept so that "5GS to EPS" and "EPS to 5GS"
    differ (their order is checked, see _in_order)."""
    out = []
    for w in _WORD_RE.findall(text or ""):
        for part in w.replace("-", " ").replace("/", " ").split():
            lw = part.lower()
            if lw in _MATCH_STOP:
                continue
            # plural of an acronym ("AFs", "NFs") is the acronym
            if len(part) >= 3 and part[:-1].isupper() and part.endswith("s"):
                lw = lw[:-1]
            if lw in ("to", "from") or (part.isupper() and len(part) >= 2) or lw not in _STOP:
                out.append(_stem(lw))
    return out


def _in_order(title: list[str], question: list[str]) -> bool:
    if "to" not in title and "from" not in title:
        return True
    pos, i = [], 0
    for t in title:
        try:
            i = question.index(t, i) + 1
        except ValueError:
            return False
    return True


def _title_of(session, hid: str) -> str:
    return _titles.get(hid, "")


def _main_root(session, question: str) -> str | None:
    """The heading of the first cited spec that the question names as its procedure.

    Pass 1 looks at the ~12 words right before that citation ("the 5G AKA procedure of TS 33.501",
    "the MO SMS over NAS procedure for a UE in CM-IDLE of TS 23.502"); pass 2, used only when pass 1
    finds nothing, looks at the whole question ("In AKMA as specified in TS 33.535, consider the
    derivation of the AKMA anchor key ..."). A heading qualifies when >= 3/4 of its title words
    (stemmed) are present and title order is kept for to/from; then near-complete coverage (>= 0.9)
    with >= 2 words, then the biggest clause subtree. Root accuracy, dev / first test set (the
    latter used for tuning after its own run): v1 30/33 and 30/47, this 32/33 and 44/47.
    """
    m = _SPEC_RE.search(question)
    if not m:
        return None
    heads = _headings(session, f"ts_{m.group(1)}_{m.group(2)}")
    for words in (_qtoks(question[:m.start()])[-12:], _qtoks(question)):
        first = {}
        for i, t in enumerate(words):
            first.setdefault(t, i)
        best = None
        for hid, _, n in heads:
            tt = _qtoks(_title_of(session, hid))
            if len(set(tt)) < 2 and not (len(tt) == 1 and tt[0] in first and len(words) <= 12):
                continue
            hit = [t for t in set(tt) if t in first]
            cover = len(hit) / len(set(tt)) if tt else 0
            if cover < 0.75 or not hit or not _in_order(tt, words):
                continue
            # restatements (interworking / I-SMF chapters, variant deltas) match the name as well
            # as the procedure's own clause; the biggest clause subtree is the procedure itself
            key = (cover >= 0.9, len(hit) >= 2, n)
            if best is None or key > best[0]:
                best = (key, hid)
        if best:
            return best[1]
    return None


def _path(hid: str) -> list[str]:
    out, h, seen = [], hid, set()
    while h and h not in seen:
        seen.add(h)
        out.append(_titles.get(h, ""))
        h = _parents.get(h)
    return [t for t in reversed(out) if t]


def _shortlist(session, question: str, spec: str, k: int = ROOT_SHORTLIST) -> list[str]:
    """Headings of `spec` ranked by question-word overlap with the heading AND its ancestors' titles,
    idf-weighted over the spec's headings. Ancestors matter because a flow's own clause is often
    titled only by its variant ("Non-roaming and Roaming with Local Breakout") under a parent that
    carries the procedure name."""
    heads = _headings(session, spec)
    qt = set(_qtoks(question))
    tts = {hid: set(_qtoks(_titles.get(hid, ""))) for hid, _, _ in heads}
    df: dict[str, int] = {}
    for tt in tts.values():
        for t in tt:
            df[t] = df.get(t, 0) + 1
    n = len(tts) or 1
    idf = lambda t: math.log(1 + n / (1 + df.get(t, 0)))
    scored = []
    for hid, _, size in heads:
        tt = tts[hid]
        anc = set()
        h = _parents.get(hid)
        while h and h in tts and len(anc) < 60:
            anc |= tts[h]
            h = _parents.get(h)
        own = sum(idf(t) for t in tt & qt)
        if not own and not (anc & qt):
            continue
        sc = own + 0.5 * sum(idf(t) for t in (anc - tt) & qt) - 0.3 * sum(idf(t) for t in tt - qt)
        scored.append((sc, size, hid))
    scored.sort(reverse=True)
    return [hid for _, _, hid in scored[:k]]


_ROOT_PROMPT = """You are given a question about a 3GPP call flow and candidate clauses of {spec}, each as \
spec and clause number | heading path (ancestors > title).

Pick the ONE clause whose subtree describes the flow the question asks about.
- Prefer the clause that specifies the procedure itself (its information flow / steps), not a clause \
that only mentions or reuses it (interworking deltas, I-SMF or roaming variants, service definitions), \
unless the question asks about that variant.
- If the question spans several variants that are sibling clauses (e.g. non-roaming and home-routed), \
pick their common parent clause.
- If no candidate is the flow, answer NONE.

Question: {question}

Candidates:
{cands}

Answer only with JSON: {{"clause": "<spec and clause number exactly as listed, or NONE>"}}"""


def _llm_root(session, question: str, llm, model: str) -> str | None:
    """The generator picks the main root from a shortlist of the first cited spec's heading tree.
    Falls back to the lexical root when the reply names no candidate. Root accuracy on the 211 seen
    questions: lexical 168, this 193. Offering every cited spec's headings measured worse (182): the
    stage-3 specs a question also cites lure the pick away from the stage-2 flow."""
    specs = list(dict.fromkeys(f"ts_{a}_{b}" for a, b in _SPEC_RE.findall(question)))[:1]
    if not specs:
        return None
    per = ROOT_SHORTLIST // len(specs)
    cands = [h for sp in specs for h in _shortlist(session, question, sp, per)]
    lex = _main_root(session, question)
    if lex and lex not in cands:
        cands.append(lex)
    if not cands:
        return None
    label = lambda h: next(f"TS {sp[3:5]}.{sp[6:]} {h[len(sp) + 1:]}" for sp in specs + [h.rsplit("_", 1)[0]]
                           if h.startswith(sp + "_"))
    by_label = {label(h): h for h in cands}
    lines = "\n".join(f"{lb} | {' > '.join(_path(h))}"
                      for lb, h in sorted(by_label.items(), key=lambda kv: _clause_key(kv[1])))
    names = ", ".join(f"TS {sp[3:5]}.{sp[6:]}" for sp in specs)
    prompt = _ROOT_PROMPT.format(spec=names, question=question, cands=lines)
    try:
        raw = llm.generate(prompt, model=model, format="json", think=False, num_predict=60)
        # cloud models wrap the JSON in a ``` fence even with format=json
        pick = str(json.loads(raw[raw.find("{"): raw.rfind("}") + 1]).get("clause") or "").strip()
    except Exception:
        return lex
    pick = re.sub(r"\s+", " ", pick.removeprefix("clause ").strip().rstrip("."))
    if pick in by_label:
        return by_label[pick]
    tail = [h for lb, h in by_label.items() if lb.endswith(" " + pick)]
    if len(tail) == 1:
        return tail[0]
    return lex if pick.upper() != "NONE" else None


def _clause_key(hid: str):
    return [(0, int(x), "") if x.isdigit() else (1, 0, x) for x in re.split(r"[._]", hid)]


def _named_roots(session, question: str, specs: list[str]) -> set[str]:
    words = _WORD_RE.findall(question)
    grams = set()
    for i in range(len(words)):
        for k in range(2, 9):
            g = frozenset(_toks(" ".join(words[i:i + k])))
            if len(g) >= 2:
                grams.add(g)
    out = set()
    for sp in specs:
        for hid, tt, _ in _heads_or_empty(session, sp):
            if frozenset(tt) in grams:
                out.add(hid)
    return out


def _heads_or_empty(session, spec):
    try:
        return _headings(session, spec)
    except Exception:
        return []


def _subtree(session, roots: list[str]) -> list[dict]:
    if not roots:
        return []
    return session.run(
        "UNWIND $roots AS r MATCH (c:Chunk) WHERE c.chunk_id = r OR c.chunk_id STARTS WITH r + 'p' "
        "OR c.chunk_id STARTS WITH r + '.' RETURN DISTINCT c.chunk_id AS chunk_id, c.content AS content, "
        "c.spec_id AS spec_id, c.section_title AS section, c.chunk_type AS chunk_type",
        roots=[r[:-2] if r.endswith(".0") else r for r in roots]).data()


_GRAPH_CLOSURE = """
UNWIND $roots AS root
MATCH (r:Procedure) WHERE r.proc_id = root OR r.proc_id STARTS WITH root + '.'
MATCH (q:Procedure)-[:SUB_PROCEDURE_OF*0..2]->(r)
WITH collect(DISTINCT q) AS ps
CALL (ps) {
  UNWIND ps AS q MATCH (q)-[:HAS_CLAUSE]->(c:Chunk) RETURN c, 'main' AS role
  UNION
  UNWIND ps AS q MATCH (q)-[:USES_OPERATION]->(:ServiceOperation)-[:DEFINED_IN]->(c:Chunk) RETURN c, 'op_def' AS role
  UNION
  UNWIND ps AS q MATCH (q)-[:INVOKES]->(:Procedure)-[:HAS_CLAUSE {role: 'main'}]->(c:Chunk) RETURN c, 'invoked' AS role
  UNION
  UNWIND ps AS q MATCH (q)-[:SAME_PROCEDURE_AS {kind: 'stage3'}]-(:Procedure)-[:HAS_CLAUSE {role: 'main'}]->(c:Chunk)
  RETURN c, 'stage3' AS role
  UNION
  UNWIND ps AS q MATCH (q)-[:USES_MESSAGE]->(:Message)-[:DEFINED_IN]->(c:Chunk) RETURN c, 'msg_def' AS role
}
RETURN DISTINCT c.chunk_id AS chunk_id, c.content AS content, c.spec_id AS spec_id,
       c.section_title AS section, c.chunk_type AS chunk_type, role
"""


def _graph_closure(session, main: str, named: list[str]) -> list[tuple[str, list[dict]]]:
    """The closure read off the Procedure layer: the main procedure's own clauses and sub-procedures,
    what it invokes, the operations / messages it uses and its stage-3 counterpart; named
    procedures contribute their own clauses."""
    base = lambda r: r[:-2] if r.endswith(".0") else r
    rows = session.run(_GRAPH_CLOSURE, roots=[base(main)]).data()
    by_role: dict[str, list[dict]] = {}
    for r in rows:
        by_role.setdefault(r["role"], []).append(r)
    named_rows = session.run(
        "UNWIND $roots AS root MATCH (r:Procedure) WHERE r.proc_id = root OR r.proc_id STARTS WITH root + '.' "
        "MATCH (r)-[:HAS_CLAUSE {role: 'main'}]->(c:Chunk) RETURN DISTINCT c.chunk_id AS chunk_id, "
        "c.content AS content, c.spec_id AS spec_id, c.section_title AS section, c.chunk_type AS chunk_type",
        roots=[base(n) for n in named]).data() if named else []
    order = ["main", "op_def", "named", "invoked", "stage3", "msg_def"]
    by_role["named"] = named_rows
    return [(role, by_role.get(role, [])) for role in order]


def _heading_chunks(session, hids: list[str], cap: int = 6) -> list[dict]:
    """The text of a heading: its own clause (+ parts), else its direct child clauses."""
    out = []
    for hid in hids:
        rows = session.run(
            "MATCH (c:Chunk) WHERE c.chunk_id = $h OR c.chunk_id STARTS WITH $h + 'p' "
            "RETURN c.chunk_id AS chunk_id, c.content AS content, c.spec_id AS spec_id, "
            "c.section_title AS section, c.chunk_type AS chunk_type", h=hid).data()
        if not rows:
            depth = hid.count(".") + 1
            rows = [r for r in session.run(
                "MATCH (c:Chunk) WHERE c.chunk_id STARTS WITH $h + '.' RETURN c.chunk_id AS chunk_id, "
                "c.content AS content, c.spec_id AS spec_id, c.section_title AS section, c.chunk_type AS chunk_type "
                "ORDER BY c.chunk_id", h=hid).data() if r["chunk_id"].count(".") == depth][:cap]
        out += rows
    return out


def _name_links(session, main_tree: list[dict], question: str, exclude: set[str]) -> tuple[list[dict], list[dict]]:
    """(named_in_text, cited_spec) chunks.

    named_in_text: a heading of the main spec or of a spec the flow cites whose full title (>= 3 words,
      not generic, not an entity name) is written out in the flow's text ("... performs the UE Policy
      Association Establishment procedure") — the flow names a clause it does not cite by number.
    cited_spec: in a spec the flow cites only as a whole ("as specified in TS 24.501"), the headings
      whose title the QUESTION covers (>= 3/4 of its words, >= 2 words) — typically the stage-3
      counterpart the question's scope asks for.
    """
    if not main_tree:
        return [], []
    tree_ids = [c["chunk_id"] for c in main_tree]
    main_spec = main_tree[0]["spec_id"]
    cited = {r["sp"] for r in session.run(
        "MATCH (c:Chunk)-[:REFERENCES_SPEC]->(d:Document) WHERE c.chunk_id IN $t RETURN DISTINCT d.spec_id AS sp",
        t=tree_ids)}
    text = f" {_raw(' '.join(c.get('content') or '' for c in main_tree))} "
    qt = _qtoks(question)
    first = {t for t in qt}
    named, by_q = [], []
    for sp in [main_spec] + sorted(cited - {main_spec}):
        heads = _heads_or_empty(session, sp)
        for hid, _, _ in heads:
            if hid in exclude:
                continue
            title = _title_of(session, hid)
            raw = _raw(title)
            words = raw.split()
            if len(words) >= 3 and raw not in _GENERIC and words[-1] not in _ENTITY_HEADS \
                    and f" {raw} " in text:
                named.append(hid)
            elif sp != main_spec:
                tt = set(_qtoks(title))
                if len(tt) >= 2 and raw not in _GENERIC:
                    hit = len(tt & first)
                    if hit >= 2 and hit / len(tt) >= 0.75:
                        by_q.append(hid)
    # most specific first: a heading inside another selected heading makes the outer one redundant
    def _prune(hids):
        hs = sorted(set(hids), key=len, reverse=True)
        keep = []
        for h in hs:
            if not any(k.startswith(h + ".") for k in keep):
                keep.append(h)
        return keep
    return _heading_chunks(session, _prune(named)[:12]), _heading_chunks(session, _prune(by_q)[:6])


def rank_closure(question: str, ce_ranked: list[dict]) -> list[dict]:
    """Order the closure by (pool-local BM25 of the question) + (cross-encoder rank), each min-max
    normalised. The cross-encoder reads only a clause's first 512 tokens, and a flow clause opens
    with context, so alone it buries the clauses the question asks about; BM25 over the whole
    clause, restricted to the closure, catches them. R@5 inside the closure, dev / test1 / test2:
    cross-encoder 0.526 / 0.335 / 0.396 -> this 0.598 / 0.402 / 0.411 (passage-max cross-encoder,
    title overlap and role priors were tried and were not better on all three)."""
    if len(ce_ranked) < 2:
        return ce_ranked
    def toks(t):
        return [w for w in re.findall(r"[a-z0-9_']+", (t or "").lower()) if w not in _STOP]
    docs = [toks((c.get("section") or "") + " " + (c.get("content") or "")) for c in ce_ranked]
    n = len(docs)
    avg = sum(map(len, docs)) / n or 1.0
    df: dict[str, int] = {}
    for dd in docs:
        for w in set(dd):
            df[w] = df.get(w, 0) + 1
    qt = set(toks(question))
    bm = []
    for dd in docs:
        tf: dict[str, int] = {}
        for w in dd:
            tf[w] = tf.get(w, 0) + 1
        sc = 0.0
        for w in qt:
            if w in tf:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                sc += idf * tf[w] * 2.2 / (tf[w] + 1.2 * (0.25 + 0.75 * len(dd) / avg))
        bm.append(sc)
    lo, hi = min(bm), max(bm)
    bm = [(v - lo) / (hi - lo) if hi > lo else 0.0 for v in bm]
    ce = [1 - i / (n - 1) for i in range(n)]
    order = sorted(range(n), key=lambda i: -(bm[i] + ce[i]))
    return [ce_ranked[i] for i in order]


def fetch_flow_closure(driver, question: str, llm=None, model: str | None = None) -> list[dict]:
    try:
        with driver.session() as s:
            if FLOW_ROOT == "llm" and llm is not None and model:
                main = _llm_root(s, question, llm, model)
            else:
                main = _main_root(s, question)
            if not main:
                return []
            specs = sorted({f"ts_{a}_{b}" for a, b in _SPEC_RE.findall(question)})
            named = sorted(_named_roots(s, question, specs) - {main})
            if FLOW_CLOSURE_SOURCE == "graph":
                groups = _graph_closure(s, main, named)
                return _assemble(groups, main)
            main_tree = _subtree(s, [main])
            named_tree = _subtree(s, named)
            tree_ids = [c["chunk_id"] for c in main_tree + named_tree]
            refs = s.run(
                "MATCH (c:Chunk)-[:REFERENCES_CHUNK]->(b:Chunk) WHERE c.chunk_id IN $t "
                "RETURN DISTINCT b.chunk_id AS chunk_id, b.content AS content, b.spec_id AS spec_id, "
                "b.section_title AS section, b.chunk_type AS chunk_type",
                t=[c["chunk_id"] for c in main_tree]).data()
            ops = s.run(
                "MATCH (c:Chunk)-[:DESCRIBES_OPERATION]->(o:ServiceOperation)<-[:DESCRIBES_OPERATION]-(d:Chunk) "
                "WHERE c.chunk_id IN $t AND d.section_title CONTAINS o.name "
                "RETURN DISTINCT d.chunk_id AS chunk_id, d.content AS content, d.spec_id AS spec_id, "
                "d.section_title AS section, d.chunk_type AS chunk_type", t=tree_ids).data()
            if FLOW_NAME_LINKS:
                base = main[:-2] if main.endswith(".0") else main
                own = {base} | {n[:-2] if n.endswith(".0") else n for n in named}
                in_text, by_spec = _name_links(s, main_tree, question, own)
            else:
                in_text, by_spec = [], []
    except Exception:
        return []
    return _assemble((("main", main_tree), ("op_def", ops), ("named", named_tree), ("ref", refs),
                      ("named_in_text", in_text), ("cited_spec", by_spec)), main)


def _assemble(groups, main: str) -> list[dict]:
    out, seen = [], set()
    for role, rows in groups:
        for r in rows:
            cid = r["chunk_id"]
            if cid in seen or not (r.get("content") or "").strip():
                continue
            seen.add(cid)
            out.append({**r, "score": ROLE_SCORE[role], "via_flow_closure": True, "flow_role": role,
                        "flow_root": main, "parent_of_anchor": False, "soft_anchor": True})
    return out[:MAX_POOL]
