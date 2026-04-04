from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from agent_memory.constants import DEFAULT_DB_PATH, DIMENSION

_DDL = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    project_path TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now', 'localtime'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    chunk_text,
    content='memories',
    content_rowid='id',
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, chunk_text) VALUES (new.id, new.chunk_text);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, chunk_text) VALUES ('delete', old.id, old.chunk_text);
END;

CREATE INDEX IF NOT EXISTS idx_memories_project_path ON memories(project_path);
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_session_id ON memories(session_id);
"""


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)

    import sqlite_vec

    conn.load_extension(sqlite_vec.loadable_path())
    conn.enable_load_extension(False)

    conn.executescript(_DDL)
    # vec0 table must be created after sqlite-vec is loaded
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
            memory_id INTEGER PRIMARY KEY,
            embedding FLOAT[{DIMENSION}]
        )
        """
    )
    conn.commit()
    return conn


def session_exists(conn: sqlite3.Connection, session_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM memories WHERE session_id = ? LIMIT 1", (session_id,)
    ).fetchone()
    return row is not None


def save_memories(
    conn: sqlite3.Connection,
    session_id: str,
    project_path: str,
    chunks: list,
    embeddings: NDArray[np.float32],
) -> None:
    for chunk, embedding in zip(chunks, embeddings):
        cursor = conn.execute(
            "INSERT INTO memories (session_id, chunk_index, chunk_text, project_path) VALUES (?, ?, ?, ?)",
            (session_id, chunk.chunk_index, chunk.chunk_text, project_path),
        )
        memory_id = cursor.lastrowid
        conn.execute(
            "INSERT INTO memories_vec (memory_id, embedding) VALUES (?, ?)",
            (memory_id, embedding.astype(np.float32).tobytes()),
        )
    conn.commit()
