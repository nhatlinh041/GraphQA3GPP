"""
SQLite connection helpers for chat history persistence.

Single-user dev tool — no connection pool; each op opens/closes a connection via
context manager. Uses stdlib `sqlite3` (sync); caller must wrap it in
`asyncio.to_thread` on the async path to avoid blocking the event loop.

Schema is applied idempotently in `init_db()` at FastAPI startup.
"""
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# Default DB file lives next to rag-engine/ code. Override via CHAT_DB_PATH env
# (for tests or to move onto a different volume mount).
DB_PATH = Path(
    os.getenv("CHAT_DB_PATH")
    or Path(__file__).resolve().parent.parent / "data" / "chat.db"
)

# WAL: writer doesn't block reader → SSE response can flush the final msg while
# another request reads history. busy_timeout 5s guards multi-tab lock contention.
_SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS conversations (
  id          TEXT PRIMARY KEY,
  created_at  INTEGER NOT NULL,
  updated_at  INTEGER NOT NULL,
  title       TEXT
);

CREATE TABLE IF NOT EXISTS messages (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
  content         TEXT NOT NULL,
  thinking        TEXT,
  stages_json     TEXT,
  sources_json    TEXT,
  started_at      INTEGER,
  created_at      INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conv_created
  ON messages(conversation_id, created_at);
"""


def init_db() -> None:
    """Create DB file + schema if missing. Idempotent — safe to call every startup."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(_SCHEMA_SQL)
        conn.commit()


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Context manager opening/closing a short connection for one CRUD op.
    `row_factory = sqlite3.Row` makes fetchall/fetchone return dict-like rows."""
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    conn.row_factory = sqlite3.Row
    # Enable foreign_keys per connection (PRAGMA scope = connection, not persisted)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()
