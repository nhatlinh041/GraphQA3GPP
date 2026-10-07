"""
Neo4j schema introspection — pulls live KG structure (labels, relationships, property
samples, populated counts) so the LLM Cypher generator can write queries against
what's actually in the database, not stale documentation.
"""
from neo4j import Driver


class SchemaIntrospector:
    """Cache-on-first-use schema reader. Refresh manually via .invalidate()."""

    def __init__(self, driver: Driver):
        self._driver = driver
        self._cached: str | None = None
        self._cached_dict: dict | None = None

    def invalidate(self) -> None:
        self._cached = None
        self._cached_dict = None

    def as_text(self) -> str:
        """Markdown-formatted schema for embedding in an LLM prompt. Cached."""
        if self._cached is None:
            self._cached = self._render(self.as_dict())
        return self._cached

    def as_dict(self) -> dict:
        """Structured schema (used by the trail UI when expanded)."""
        if self._cached_dict is None:
            self._cached_dict = self._collect()
        return self._cached_dict

    # ----- collection -----

    def _collect(self) -> dict:
        labels = self._labels()
        rels = self._relationship_types()

        nodes: list[dict] = []
        for label in labels:
            count = self._safe_scalar(f"MATCH (n:`{label}`) RETURN count(n) AS c", "c", 0)
            # keys(n) already returns a list; pull it directly via _safe_scalar.
            props = self._safe_scalar(
                f"MATCH (n:`{label}`) WITH n LIMIT 1 RETURN keys(n) AS k", "k", []
            ) or []
            nodes.append({"label": label, "count": count, "properties": props})

        relationships: list[dict] = []
        for rt in rels:
            # Sample one edge to learn the typical (start_label, end_label) for this rel
            sample = self._safe_record(
                f"MATCH (a)-[r:`{rt}`]->(b) "
                "RETURN labels(a) AS a, labels(b) AS b, count(*) AS c LIMIT 1"
            )
            count = self._safe_scalar(
                f"MATCH ()-[r:`{rt}`]->() RETURN count(r) AS c", "c", 0
            )
            if sample:
                a_labels = sample.get("a") or []
                b_labels = sample.get("b") or []
                relationships.append({
                    "type": rt,
                    "count": count,
                    "from": a_labels[0] if a_labels else "?",
                    "to": b_labels[0] if b_labels else "?",
                })
            else:
                relationships.append({"type": rt, "count": count, "from": "?", "to": "?"})

        return {"nodes": nodes, "relationships": relationships}

    # ----- rendering -----

    @staticmethod
    def _render(schema: dict) -> str:
        lines: list[str] = []
        lines.append("Node labels (with property names and node count):")
        for n in schema["nodes"]:
            props = ", ".join(n["properties"]) if n["properties"] else "—"
            lines.append(f"  ({n['label']}) — {n['count']} nodes — properties: {props}")
        lines.append("")
        lines.append("Relationships ((from)-[:TYPE]->(to), with edge count):")
        for r in schema["relationships"]:
            lines.append(f"  ({r['from']})-[:{r['type']}]->({r['to']}) — {r['count']} edges")
        return "\n".join(lines)

    # ----- low-level helpers -----

    def _labels(self) -> list[str]:
        return self._safe_list("CALL db.labels() YIELD label RETURN label", "label", [])

    def _relationship_types(self) -> list[str]:
        return self._safe_list(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType",
            "relationshipType",
            [],
        )

    def _safe_scalar(self, cypher: str, key: str, default):
        try:
            with self._driver.session() as s:
                rec = s.run(cypher).single()
                return rec[key] if rec is not None else default
        except Exception:
            return default

    def _safe_list(self, cypher: str, key: str, default):
        try:
            with self._driver.session() as s:
                return [r[key] for r in s.run(cypher)]
        except Exception:
            return default

    def _safe_record(self, cypher: str) -> dict | None:
        try:
            with self._driver.session() as s:
                rec = s.run(cypher).single()
                return dict(rec) if rec is not None else None
        except Exception:
            return None
