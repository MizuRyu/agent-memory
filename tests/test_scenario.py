"""End-to-end scenario test covering all agent-memory features.

Flow: transcript → chunker → embedder → store → search
Verifies: chunking, embedding, storage, FTS5, vector search, RRF, project weight, time decay
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from agent_memory.chunker import chunk_transcript
from agent_memory.search import hybrid_search
from agent_memory.store import save_memories


def _make_transcript(entries: list[dict]) -> Path:
    """Write a fake JSONL transcript to a temp file."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
    for entry in entries:
        tmp.write(json.dumps(entry, ensure_ascii=False) + "\n")
    tmp.close()
    return Path(tmp.name)


def _user_msg(text: str, session_id: str = "sess-001") -> dict:
    return {
        "type": "user",
        "message": {"role": "user", "content": text},
        "sessionId": session_id,
        "uuid": "u1",
        "timestamp": "2026-04-01T10:00:00Z",
    }


def _assistant_msg(text: str, session_id: str = "sess-001") -> dict:
    return {
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
        "sessionId": session_id,
        "uuid": "a1",
        "timestamp": "2026-04-01T10:00:01Z",
    }


def _meta_msg(text: str, session_id: str = "sess-001") -> dict:
    return {
        "type": "user",
        "isMeta": True,
        "message": {"role": "user", "content": text},
        "sessionId": session_id,
    }


def _tool_use_msg(session_id: str = "sess-001") -> dict:
    return {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}],
        },
        "sessionId": session_id,
    }


def test_full_scenario(tmp_db, fake_embedder):
    conn, db_path = tmp_db

    # ============================
    # Phase 1: Chunker
    # ============================

    print("\n========== Phase 1: Chunker ==========")

    transcript_path = _make_transcript([
        {"type": "file-history-snapshot", "snapshot": {}},
        _meta_msg("This is a skill injection"),
        _user_msg("OAuth2.0とAPIキー認証、どちらが適切ですか？"),
        _assistant_msg("このユースケースではOAuth2.0を推奨します。理由はセキュリティと拡張性です。"),
        _tool_use_msg(),
        _user_msg("SQLiteのパフォーマンスチューニングについて教えてください"),
        _assistant_msg("主要なチューニングポイントは、WALモード、インデックス設計、PRAGMA設定です。"),
        _user_msg("Pythonのエラーハンドリングのベストプラクティスは？"),
        _assistant_msg("基本方針として、具体的な例外をキャッチし、適切なレベルでハンドリングすることが重要です。"),
    ])

    result = chunk_transcript(transcript_path)

    print(f"Session ID: {result.session_id}")
    print(f"Chunks: {len(result.chunks)}")
    for c in result.chunks:
        print(f"  [{c.chunk_index}] {c.chunk_text[:80]}...")

    assert result.session_id == "sess-001"
    assert len(result.chunks) == 3
    assert "OAuth2.0" in result.chunks[0].chunk_text
    assert "SQLite" in result.chunks[1].chunk_text
    assert "エラーハンドリング" in result.chunks[2].chunk_text
    assert [c.chunk_index for c in result.chunks] == [0, 1, 2]

    # ============================
    # Phase 2: Embedding
    # ============================

    print("\n========== Phase 2: Embedding ==========")

    embeddings = fake_embedder.embed_documents([c.chunk_text for c in result.chunks])

    print(f"Shape: {embeddings.shape}")
    for i in range(len(result.chunks)):
        norm = float(np.linalg.norm(embeddings[i]))
        print(f"  [{i}] L2 norm: {norm:.6f}  first 5 dims: {embeddings[i][:5]}")

    assert embeddings.shape == (3, 768)
    for i in range(3):
        assert abs(float(np.linalg.norm(embeddings[i])) - 1.0) < 1e-5

    # ============================
    # Phase 3: Store
    # ============================

    print("\n========== Phase 3: Store ==========")

    project_alpha = "/Users/test/project-alpha"
    save_memories(conn, result.session_id, project_alpha, result.chunks, embeddings)

    count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    fts_count = conn.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
    vec_count = conn.execute("SELECT COUNT(*) FROM memories_vec").fetchone()[0]

    print(f"memories:     {count} rows")
    print(f"memories_fts: {fts_count} rows")
    print(f"memories_vec: {vec_count} rows")

    assert count == 3
    assert fts_count == 3
    assert vec_count == 3

    paths = conn.execute("SELECT DISTINCT project_path FROM memories").fetchall()
    assert paths == [(project_alpha,)]

    # ============================
    # Phase 4: Add second project data
    # ============================

    print("\n========== Phase 4: Add project-beta ==========")

    from agent_memory.chunker import Chunk

    chunks_beta = [
        Chunk(chunk_index=0, chunk_text="User: Rustの認証設計について\n\nAssistant: RustではOAuth2.0ライブラリとしてoxide-authがあります。"),
    ]
    embeddings_beta = fake_embedder.embed_documents([c.chunk_text for c in chunks_beta])
    project_beta = "/Users/test/project-beta"
    save_memories(conn, "sess-002", project_beta, chunks_beta, embeddings_beta)

    total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    print(f"Total memories: {total} rows (alpha: 3, beta: 1)")
    assert total == 4

    # ============================
    # Phase 5: Hybrid search + project weight
    # ============================

    print("\n========== Phase 5: Search \"認証\" (project=alpha) ==========")

    query = "認証"
    query_vec = fake_embedder.embed_query(query)

    results = hybrid_search(conn, query, query_vec, project_path=project_alpha, top_n=10)
    assert len(results) > 0

    for i, r in enumerate(results, 1):
        pj = r.project_path.split("/")[-1]
        print(f"  [{i}] score={r.score:.6f}  project={pj}  text={r.chunk_text[:70]}...")

    alpha_results = [r for r in results if r.project_path == project_alpha]
    beta_results = [r for r in results if r.project_path == project_beta]

    if alpha_results and beta_results:
        print(f"  → alpha best: {alpha_results[0].score:.6f} > beta best: {beta_results[0].score:.6f} ✓ (project weight)")
        assert alpha_results[0].score > beta_results[0].score

    # ============================
    # Phase 6: Time decay
    # ============================

    print("\n========== Phase 6: Time decay (id=1 → 60 days old) ==========")

    old_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%S")
    conn.execute("UPDATE memories SET created_at = ? WHERE id = 1", (old_date,))
    conn.commit()

    results_decay = hybrid_search(conn, query, query_vec, project_path=project_alpha, top_n=10)

    for i, r in enumerate(results_decay, 1):
        tag = " ← 60 days old" if r.memory_id == 1 else ""
        pj = r.project_path.split("/")[-1]
        print(f"  [{i}] score={r.score:.6f}  id={r.memory_id}  project={pj}{tag}")

    old_result = next((r for r in results_decay if r.memory_id == 1), None)
    if old_result:
        fresh_alpha = [r for r in results_decay if r.project_path == project_alpha and r.memory_id != 1]
        if fresh_alpha:
            print(f"  → fresh: {fresh_alpha[0].score:.6f} > old(60d): {old_result.score:.6f} ✓ (time decay)")
            assert old_result.score < fresh_alpha[0].score

    # ============================
    # Phase 7: Search without project weighting
    # ============================

    print("\n========== Phase 7: Search without project weighting ==========")

    results_no_project = hybrid_search(conn, query, query_vec, project_path=None, top_n=10)
    assert len(results_no_project) > 0

    for i, r in enumerate(results_no_project, 1):
        pj = r.project_path.split("/")[-1]
        print(f"  [{i}] score={r.score:.6f}  project={pj}  text={r.chunk_text[:70]}...")

    beta_no_pj = [r for r in results_no_project if r.project_path == project_beta]
    if beta_results and beta_no_pj:
        print(f"  → beta (no weight): {beta_no_pj[0].score:.6f} > beta (weighted): {beta_results[0].score:.6f} ✓")
        assert beta_no_pj[0].score > beta_results[0].score

    # ============================
    # Phase 8: Verify result structure
    # ============================

    print("\n========== Phase 8: Result structure check ==========")

    for r in results_no_project:
        assert isinstance(r.memory_id, int)
        assert isinstance(r.chunk_text, str)
        assert isinstance(r.score, float)
        assert isinstance(r.project_path, str)
        assert isinstance(r.created_at, str)
        assert isinstance(r.session_id, str)
        assert r.score > 0

    print("All fields valid ✓")

    # Cleanup
    Path(transcript_path).unlink(missing_ok=True)
    print("\n========== All phases passed ✓ ==========")
