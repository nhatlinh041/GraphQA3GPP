"""
ConversationStore — CRUD for chat history.

All methods are synchronous (sqlite3 stdlib). Async callers must wrap via
`asyncio.to_thread(...)` so they don't block FastAPI's event loop.

Stages are stored as RAW JSON (no server-side compaction) — single-user dev
tool, a few MB per turn fits comfortably in a SQLite TEXT column. Trade-off:
simpler, no need to port `compactStagesForStorage` from TS to Python.
"""
import json
import time
import uuid
from typing import Any

from .db import connect


def _now_ms() -> int:
    """Epoch milliseconds — matches frontend `Date.now()`."""
    return int(time.time() * 1000)


class ConversationStore:
    def create(self) -> str:
        """Create a new conversation with a UUIDv4 id; return the id."""
        cid = str(uuid.uuid4())
        now = _now_ms()
        with connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, created_at, updated_at, title) "
                "VALUES (?, ?, ?, NULL)",
                (cid, now, now),
            )
            conn.commit()
        return cid

    def exists(self, cid: str) -> bool:
        """Check whether id exists in the DB — used to validate request body."""
        with connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (cid,)
            ).fetchone()
        return row is not None

    def get(self, cid: str) -> dict | None:
        """Return conversation metadata + all messages.
        Return None if cid does not exist."""
        with connect() as conn:
            conv = conn.execute(
                "SELECT id, created_at, updated_at, title FROM conversations WHERE id = ?",
                (cid,),
            ).fetchone()
            if conv is None:
                return None
            rows = conn.execute(
                "SELECT role, content, thinking, stages_json, sources_json, started_at "
                "FROM messages WHERE conversation_id = ? ORDER BY created_at ASC, id ASC",
                (cid,),
            ).fetchall()
        # Parse JSON in Python so the frontend gets the right shape (stages: array | null)
        messages = [
            {
                "role": r["role"],
                "content": r["content"],
                "thinking": r["thinking"],
                "stages": json.loads(r["stages_json"]) if r["stages_json"] else None,
                "sources": json.loads(r["sources_json"]) if r["sources_json"] else None,
                "startedAt": r["started_at"],
            }
            for r in rows
        ]
        return {
            "id": conv["id"],
            "created_at": conv["created_at"],
            "updated_at": conv["updated_at"],
            "title": conv["title"],
            "messages": messages,
        }

    def append_user(self, cid: str, content: str, started_at: int) -> None:
        """Persist user message IMMEDIATELY on query — survives a refresh mid-stream."""
        now = _now_ms()
        with connect() as conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, started_at, created_at) "
                "VALUES (?, 'user', ?, ?, ?)",
                (cid, content, started_at, now),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, cid)
            )
            # Auto-set title from the first prompt (truncated to 80 chars) — for the sidebar later
            conn.execute(
                "UPDATE conversations SET title = ? "
                "WHERE id = ? AND title IS NULL",
                (content[:80], cid),
            )
            conn.commit()

    def append_assistant(
        self,
        cid: str,
        content: str,
        thinking: str | None,
        stages: list[dict[str, Any]],
        sources: list[dict[str, Any]] | None,
        started_at: int,
    ) -> None:
        """Persist assistant turn after streaming finishes. stages = raw event list."""
        now = _now_ms()
        with connect() as conn:
            conn.execute(
                "INSERT INTO messages "
                "(conversation_id, role, content, thinking, stages_json, sources_json, "
                " started_at, created_at) "
                "VALUES (?, 'assistant', ?, ?, ?, ?, ?, ?)",
                (
                    cid,
                    content,
                    thinking,
                    json.dumps(stages) if stages else None,
                    json.dumps(sources) if sources else None,
                    started_at,
                    now,
                ),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, cid)
            )
            conn.commit()

    def delete(self, cid: str) -> bool:
        """ON DELETE CASCADE auto-deletes messages. Return True if cid existed."""
        with connect() as conn:
            cur = conn.execute("DELETE FROM conversations WHERE id = ?", (cid,))
            conn.commit()
            return cur.rowcount > 0
