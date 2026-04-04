from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from agent_memory.constants import (
    DECAY_HALF_LIFE_DAYS,
    PROJECT_WEIGHT_DIFF,
    PROJECT_WEIGHT_SAME,
    RRF_K,
)


@dataclass
class SearchResult:
    memory_id: int
    chunk_text: str
    score: float
    project_path: str
    created_at: str
    session_id: str


def _fts_search(
    conn: sqlite3.Connection, query: str, limit: int
) -> list[tuple[int, int]]:
    """FTS5 keyword search. Returns list of (rowid, rank_position)."""
    try:
        rows = conn.execute(
            "SELECT rowid, rank FROM memories_fts WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?",
            (f'"{query}"', limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [(row[0], i + 1) for i, row in enumerate(rows)]


def _vec_search(
    conn: sqlite3.Connection, query_embedding: NDArray[np.float32], limit: int
) -> list[tuple[int, int]]:
    """Vector similarity search. Returns list of (memory_id, rank_position)."""
    rows = conn.execute(
        "SELECT memory_id, distance FROM memories_vec WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (query_embedding.astype(np.float32).tobytes(), limit),
    ).fetchall()
    return [(row[0], i + 1) for i, row in enumerate(rows)]


def _compute_rrf(
    fts_results: list[tuple[int, int]],
    vec_results: list[tuple[int, int]],
) -> dict[int, float]:
    """Compute RRF scores from two ranked lists."""
    scores: dict[int, float] = {}
    for memory_id, rank in fts_results:
        scores[memory_id] = scores.get(memory_id, 0.0) + 1.0 / (RRF_K + rank)
    for memory_id, rank in vec_results:
        scores[memory_id] = scores.get(memory_id, 0.0) + 1.0 / (RRF_K + rank)
    return scores


def _time_decay(created_at: str, now: datetime) -> float:
    """Compute time decay factor. Half-life = 30 days."""
    try:
        dt = datetime.fromisoformat(created_at)
    except ValueError:
        return 1.0
    days = (now - dt).total_seconds() / 86400
    if days < 0:
        days = 0
    return 0.5 ** (days / DECAY_HALF_LIFE_DAYS)


def hybrid_search(
    conn: sqlite3.Connection,
    query_text: str,
    query_embedding: NDArray[np.float32],
    project_path: str | None = None,
    top_n: int = 5,
    candidate_n: int = 50,
) -> list[SearchResult]:
    fts_results = _fts_search(conn, query_text, candidate_n)
    vec_results = _vec_search(conn, query_embedding, candidate_n)

    rrf_scores = _compute_rrf(fts_results, vec_results)
    if not rrf_scores:
        return []

    # Fetch metadata for candidate memory IDs
    memory_ids = list(rrf_scores.keys())
    placeholders = ",".join("?" for _ in memory_ids)
    rows = conn.execute(
        f"SELECT id, session_id, chunk_text, project_path, created_at FROM memories WHERE id IN ({placeholders})",
        memory_ids,
    ).fetchall()

    now = datetime.now()
    results: list[SearchResult] = []
    for row in rows:
        mid, sid, text, ppath, cat = row
        rrf = rrf_scores[mid]

        pw = PROJECT_WEIGHT_SAME
        if project_path is not None and ppath != project_path:
            pw = PROJECT_WEIGHT_DIFF

        decay = _time_decay(cat, now)
        final_score = rrf * pw * decay

        results.append(
            SearchResult(
                memory_id=mid,
                chunk_text=text,
                score=final_score,
                project_path=ppath,
                created_at=cat,
                session_id=sid,
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_n]
