#!/usr/bin/env python3
"""
Standalone verifier — runs the same checks as test_output_json.py
but without pytest. Handy to run right after the pipeline finishes.

Usage:
    python document_processing/tests/verify.py [JSON_DIR]

Default JSON_DIR: 3GPP_JSON_DOC/processed_json_v6

Exit code: 0 if all pass, 1 on error.
"""
import json
import re
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON_DIR = REPO_ROOT / "3GPP_JSON_DOC" / "processed_json_v6"

SPEC_ID_RE = re.compile(r"^ts_\d{2}_\d{3}(?:-\d+)?$")
SPEC_ID_LOOSE_RE = re.compile(r"^ts_\d+_\d+(?:-\w+)?$")


# ANSI color codes for terminal output
class C:
    R = "\033[31m"
    G = "\033[32m"
    Y = "\033[33m"
    B = "\033[36m"
    X = "\033[0m"


def fail(msg: str) -> None:
    print(f"{C.R}✗{C.X} {msg}")


def ok(msg: str) -> None:
    print(f"{C.G}✓{C.X} {msg}")


def warn(msg: str) -> None:
    print(f"{C.Y}!{C.X} {msg}")


def info(msg: str) -> None:
    print(f"{C.B}→{C.X} {msg}")


def load_all(json_dir: Path) -> List[dict]:
    # _path is stashed on each dict so errors can name the offending file
    docs = []
    for p in sorted(json_dir.glob("*.json")):
        try:
            with p.open(encoding="utf-8") as f:
                docs.append({"_path": p, **json.load(f)})
        except Exception as e:
            fail(f"Cannot parse {p.name}: {e}")
    return docs


def run_checks(docs: List[dict]) -> Tuple[int, int]:
    """Run the checks. Returns (passed, failed) count."""
    passed, failed = 0, 0

    def check(name: str, violations: List, sample_n: int = 5) -> None:
        nonlocal passed, failed
        if violations:
            failed += 1
            fail(f"{name} — {len(violations)} violation(s)")
            for v in violations[:sample_n]:
                print(f"     {v}")
            if len(violations) > sample_n:
                print(f"     ... ({len(violations) - sample_n} more)")
        else:
            passed += 1
            ok(name)

    bad = [
        (d["_path"].name, d["metadata"]["specification_id"])
        for d in docs
        if "." in d["metadata"]["specification_id"]
    ]
    check("specification_id no longer in dot format", bad)

    nonconform = [
        (d["_path"].name, d["metadata"]["specification_id"])
        for d in docs
        if not SPEC_ID_RE.match(d["metadata"]["specification_id"])
    ]
    check("specification_id matches ts_NN_NNN[-P]", nonconform)

    name_mismatch = [
        (d["_path"].name, f"{d['metadata']['specification_id']}.json")
        for d in docs
        if d["_path"].name != f"{d['metadata']['specification_id']}.json"
    ]
    check("filename matches specification_id", name_mismatch)

    empty = [d["_path"].name for d in docs if not d.get("chunks")]
    check("every document has chunks", empty)

    cid_bad = []
    for d in docs:
        sid = d["metadata"]["specification_id"]
        for c in d.get("chunks", []):
            if not c.get("chunk_id", "").startswith(f"{sid}_"):
                cid_bad.append((d["_path"].name, c.get("chunk_id")))
                break
    check("chunk_id starts with spec_id", cid_bad)

    cid_dup = []
    for d in docs:
        seen = set()
        for c in d.get("chunks", []):
            cid = c["chunk_id"]
            if cid in seen:
                cid_dup.append((d["_path"].name, cid))
            seen.add(cid)
    check("chunk_id unique within each document", cid_dup)

    # target_spec must be underscore form (Bug #2 + #3)
    dot_refs = []
    for d in docs:
        for c in d.get("chunks", []):
            for ref in c.get("cross_references", {}).get("external", []):
                t = ref.get("target_spec", "")
                if "." in t:
                    dot_refs.append((d["_path"].name, c["chunk_id"], t))
    check("target_spec no longer in dot format", dot_refs)

    # Series 01-12 is GSM-era numbering and legitimately 2-digit ("GSM TS 11.14"),
    # so only a modern series (21+) with <3 digits looks truncated. Keep in sync
    # with tests/test_output_json.py::test_target_spec_no_truncation.
    # …and a short number the SOURCE ITSELF writes ("TS 38.01-1" is a typo in the
    # spec for 38.101-1) is copied faithfully, not truncated by us.
    trunc = []
    for d in docs:
        for c in d.get("chunks", []):
            content = c.get("content", "")
            for ref in c.get("cross_references", {}).get("external", []):
                t = ref.get("target_spec", "")
                m = re.match(r"^ts_(\d+)_(\d+)", t)
                if not (m and len(m.group(2)) < 3 and int(m.group(1)) > 12):
                    continue
                literal = re.compile(
                    r"\b(?:TS|TR)\s+" + re.escape(f"{m.group(1)}.{m.group(2)}") + r"\b",
                    re.IGNORECASE,
                )
                if not literal.search(content):
                    trunc.append((d["_path"].name, c["chunk_id"], t))
    check("target_spec not truncated (>=3 digits, 3GPP series)", trunc)

    # target_spec must have a valid shape (Pattern 2 bug)
    bad_shape = []
    for d in docs:
        for c in d.get("chunks", []):
            for ref in c.get("cross_references", {}).get("external", []):
                t = ref.get("target_spec", "")
                if not SPEC_ID_LOOSE_RE.match(t):
                    bad_shape.append((d["_path"].name, c["chunk_id"], t))
    check("target_spec has shape ts_X_Y", bad_shape)

    # a doc must not list itself as an external cross-reference
    self_refs = []
    for d in docs:
        sid = d["metadata"]["specification_id"]
        for c in d.get("chunks", []):
            for ref in c.get("cross_references", {}).get("external", []):
                if ref.get("target_spec") == sid:
                    self_refs.append((d["_path"].name, c["chunk_id"]))
                    break
    check("self-references do not leak into external", self_refs)

    return passed, failed


def print_summary(docs: List[dict]) -> None:
    total_chunks = sum(len(d.get("chunks", [])) for d in docs)
    total_ext = 0
    distinct = set()
    for d in docs:
        for c in d.get("chunks", []):
            for ref in c.get("cross_references", {}).get("external", []):
                total_ext += 1
                distinct.add(ref.get("target_spec", ""))

    all_sids = {d["metadata"]["specification_id"] for d in docs}
    # tolerate multi-part specs (ts_38_508 covers ts_38_508-1) when flagging orphans
    orphan = {
        t for t in distinct
        if t not in all_sids and not any(s.startswith(f"{t}-") for s in all_sids)
    }

    print()
    print(f"{C.B}=== Stats ==={C.X}")
    print(f"  Documents:        {len(docs)}")
    print(f"  Chunks:           {total_chunks}")
    print(f"  External refs:    {total_ext}")
    print(f"  Distinct targets: {len(distinct)}")
    print(f"  Orphan targets:   {len(orphan)} (no matching Document in the corpus)")
    if orphan:
        sample = sorted(orphan)[:10]
        print(f"    Sample: {sample}")


def main() -> int:
    json_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JSON_DIR
    if not json_dir.is_dir():
        fail(f"JSON dir does not exist: {json_dir}")
        return 1

    info(f"Verify: {json_dir}")
    docs = load_all(json_dir)
    if not docs:
        fail(f"No JSON files in {json_dir}")
        return 1
    info(f"Loaded {len(docs)} files")
    print()

    passed, failed = run_checks(docs)
    print_summary(docs)

    print()
    if failed:
        fail(f"{failed} check fail / {passed + failed} total")
        return 1
    ok(f"All {passed} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
