"""
TermIndex — in-memory snapshot of all `Term` nodes in the KG, built once at
startup. Replaces hard-coded NETWORK_FUNCTIONS + per-query Neo4j round-trips
with case-insensitive lookups.

Loads ~31k Terms (~6 MB) and exposes:
- abbreviation lookup     (case-insensitive)
- full_name lookup        (case-insensitive)
- full_name regex matcher (longest-first alternation)
- substring search        (substitute for the live `kg_search_terms` Cypher)
- legacy resolve()        (drop-in for the old TermResolver contract)
"""
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# Record stored in the index. Mirrors the shape returned by the old
# AdaptiveHopSearcher._tool_expand_term Cypher so downstream code is unchanged.
TermRecord = dict  # {abbreviation: str, full_name: str, primary_spec: str|None, source_specs: list[str]}


class TermIndex:
    def __init__(
        self,
        abbrev_map: dict[str, TermRecord],
        full_name_map: dict[str, str],
        full_name_re: Optional[re.Pattern],
    ):
        # UPPER-cased abbreviation → record
        self.abbrev_map = abbrev_map
        # lower-cased full_name → abbreviation (the canonical key into abbrev_map)
        self.full_name_map = full_name_map
        # Longest-first alternation regex over full_names (IGNORECASE, word-bounded).
        # None when the index is empty.
        self._full_name_re = full_name_re

    @classmethod
    def empty(cls) -> "TermIndex":
        # Graceful degrade when the KG is unreachable at startup.
        return cls(abbrev_map={}, full_name_map={}, full_name_re=None)

    # Case-insensitive abbreviation lookup. Input may be any case ("scp", "SCP", "Scp").
    def lookup_abbrev(self, token: str) -> Optional[TermRecord]:
        if not token:
            return None
        return self.abbrev_map.get(token.strip().upper())

    # Case-insensitive full-name lookup. Input may be any case.
    def lookup_full_name(self, name: str) -> Optional[TermRecord]:
        if not name:
            return None
        abbrev = self.full_name_map.get(name.strip().lower())
        if not abbrev:
            return None
        return self.abbrev_map.get(abbrev)

    # Scan the query for full_name matches. Longest-first means
    # "Service Communication Proxy" wins over "Service" or "Proxy".
    # Returns [(abbreviation, matched_full_name)] in order of appearance.
    def find_full_name_matches(self, query: str) -> list[tuple[str, str]]:
        if not self._full_name_re or not query:
            return []
        out: list[tuple[str, str]] = []
        seen: set[str] = set()
        for m in self._full_name_re.finditer(query):
            matched = m.group(0)
            abbrev = self.full_name_map.get(matched.lower())
            if not abbrev or abbrev in seen:
                continue
            seen.add(abbrev)
            out.append((abbrev, matched))
        return out

    # Substring search across abbreviation + full_name. Used by adaptive_hop's
    # `kg_search_terms` tool — replaces the live Cypher `toLower(...) CONTAINS ...`.
    def search(self, keyword: str, limit: int = 20) -> list[TermRecord]:
        if not keyword:
            return []
        kw = keyword.strip().lower()
        out: list[TermRecord] = []
        seen: set[str] = set()
        for abbrev, rec in self.abbrev_map.items():
            full_name = (rec.get("full_name") or "")
            if kw in abbrev.lower() or kw in full_name.lower():
                if abbrev in seen:
                    continue
                seen.add(abbrev)
                out.append(rec)
                if len(out) >= limit:
                    break
        return out

    # Legacy TermResolver.resolve contract: list of strings (any case) →
    # {original_token: {full_name, specs, matched_property}}. Entries
    # without a hit are simply omitted, matching the old behaviour.
    def resolve(self, entities: list[str]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for entity in entities:
            rec = self.lookup_abbrev(entity) or self.lookup_full_name(entity)
            if not rec:
                continue
            out[entity] = {
                "full_name": rec.get("full_name"),
                "specs": list(rec.get("source_specs") or []),
                "matched_property": "abbreviation",
            }
        return out


# Build a TermIndex from a list of raw records (used by tests + factory).
def build_from_records(records: list[dict]) -> TermIndex:
    abbrev_map: dict[str, TermRecord] = {}
    full_name_map: dict[str, str] = {}

    for raw in records:
        abbrev = (raw.get("abbreviation") or "").strip()
        if not abbrev:
            continue
        abbrev_upper = abbrev.upper()
        full_name = (raw.get("full_name") or "").strip()
        primary_spec = raw.get("primary_spec")
        source_specs = list(raw.get("source_specs") or [])

        record: TermRecord = {
            "abbreviation": abbrev_upper,
            "full_name": full_name,
            "primary_spec": primary_spec,
            "source_specs": source_specs,
        }
        # Source node label (Message/ServiceOperation/Concept/Parameter). Absent = :Term.
        if raw.get("node_label"):
            record["node_label"] = raw["node_label"]

        # Duplicate abbreviation: keep the entry with the larger source_specs list
        # (more data → more useful for downstream Cypher Pattern A1).
        existing = abbrev_map.get(abbrev_upper)
        if existing is None or len(source_specs) > len(existing.get("source_specs") or []):
            abbrev_map[abbrev_upper] = record

        # full_name → abbreviation. Skip empty / collisions: first abbrev wins,
        # which is fine because re-runs after a rebuild are deterministic.
        if full_name:
            key = full_name.lower()
            full_name_map.setdefault(key, abbrev_upper)

    full_name_re = _compile_full_name_regex(list(full_name_map.keys()))
    return TermIndex(abbrev_map=abbrev_map, full_name_map=full_name_map, full_name_re=full_name_re)


# Compile a single alternation regex over all full_names.
# Sort longest-first so "Service Communication Proxy" is matched before "Service".
# Returns None when there are no names (empty KG).
def _compile_full_name_regex(names_lower: list[str]) -> Optional[re.Pattern]:
    if not names_lower:
        return None
    # Longest first; tie-break alphabetically for determinism.
    sorted_names = sorted(names_lower, key=lambda n: (-len(n), n))
    escaped = [re.escape(n) for n in sorted_names]
    pattern = r"\b(" + "|".join(escaped) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


# Factory: load all Term nodes from Neo4j once. Returns TermIndex.empty() on
# any failure so the rest of the pipeline can still boot (with degraded term
# extraction).
# Entity nodes beyond :Term that a question can name verbatim. Without these the
# extractor only ever resolves :Term, so a question naming a message
# ("DEDICATED MEASUREMENT FAILURE INDICATION"), an SBI operation
# ("Nnef_ECSAddress_Create"), a table parameter or a value concept resolves
# NOTHING and the Cypher generator falls through to Pattern C — measured as the
# single biggest cause of KG misses on the factoid class. Each node's name is fed
# in as BOTH abbreviation and full_name so the existing two lookup paths
# (token match, longest-full-name regex) both work unchanged.
_ENTITY_SOURCES = (
    ("Message", "MATCH (n:Message) WHERE n.name IS NOT NULL AND n.name <> '' "
                "RETURN n.name AS name"),
    ("ServiceOperation", "MATCH (n:ServiceOperation) WHERE n.name IS NOT NULL AND n.name <> '' "
                         "RETURN n.name AS name"),
    ("Concept", "MATCH (n:Concept) WHERE n.name IS NOT NULL AND n.name <> '' "
                "RETURN n.name AS name"),
    # Parameter is DELIBERATELY NOT loaded: 78k per-chunk nodes, and on 100 factoid
    # questions loading it LOST 8 `information element` hits (the LLM switched to a
    # wrong Pattern P instead of the pattern that was working) while gaining only ~2.
    # Message/ServiceOperation/Concept are a pure gain (+8 message, +5 procedure).
)


def build_term_index(driver) -> TermIndex:
    cypher = (
        "MATCH (t:Term) "
        "WHERE t.abbreviation IS NOT NULL AND t.abbreviation <> '' "
        "RETURN t.abbreviation AS abbreviation, "
        "       t.full_name    AS full_name, "
        "       t.primary_spec AS primary_spec, "
        "       t.source_specs AS source_specs"
    )
    try:
        with driver.session() as session:
            records = [dict(r) for r in session.run(cypher)]
            n_term = len(records)
            for label, q in _ENTITY_SOURCES:
                try:
                    for r in session.run(q):
                        name = (r["name"] or "").strip()
                        if name:
                            records.append({"abbreviation": name, "full_name": name,
                                            "primary_spec": None, "source_specs": [],
                                            "node_label": label})
                except Exception as e:
                    logger.warning("TermIndex: skipping %s nodes (%s)", label, e)
    except Exception as e:
        logger.warning("TermIndex build failed (%s) — falling back to empty index", e)
        return TermIndex.empty()

    index = build_from_records(records)
    logger.info(
        "TermIndex loaded: %d abbreviations (%d Term + %d entity), %d full_names",
        len(index.abbrev_map), n_term, len(records) - n_term,
        len(index.full_name_map),
    )
    return index
