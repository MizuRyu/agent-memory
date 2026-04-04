from __future__ import annotations

import subprocess
from pathlib import Path

import typer

from agent_memory.constants import DEFAULT_DB_PATH

app = typer.Typer(name="agent-memory", help="Long-term memory for Claude Code")


@app.command()
def save(
    transcript: Path = typer.Option(..., help="Path to transcript JSONL file"),
    project: Path = typer.Option(..., help="Project directory path"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show chunks without saving"),
) -> None:
    """Save a session transcript to memory."""
    from agent_memory.chunker import chunk_transcript

    result = chunk_transcript(transcript)
    if not result.chunks:
        typer.echo("No chunks extracted from transcript.")
        return

    if dry_run:
        typer.echo(f"Session: {result.session_id}")
        typer.echo(f"Project: {project.resolve()}")
        typer.echo(f"Chunks:  {len(result.chunks)}")
        for c in result.chunks:
            typer.echo(f"\n--- chunk {c.chunk_index} ({len(c.chunk_text)} chars) ---")
            preview = c.chunk_text[:200]
            if len(c.chunk_text) > 200:
                preview += "..."
            typer.echo(preview)
        return

    from agent_memory.embedder import RuriEmbedder
    from agent_memory.store import get_connection, save_memories, session_exists

    conn = get_connection()
    if session_exists(conn, result.session_id):
        typer.echo(f"Session {result.session_id} already saved. Skipping.")
        conn.close()
        return

    embedder = RuriEmbedder()
    embeddings = embedder.embed_documents([c.chunk_text for c in result.chunks])

    save_memories(conn, result.session_id, str(project.resolve()), result.chunks, embeddings)
    conn.close()

    typer.echo(f"Saved {len(result.chunks)} chunks from session {result.session_id}")


@app.command()
def search(
    query: str = typer.Option(..., help="Search query text"),
    project: Path | None = typer.Option(None, help="Current project path for weighting"),
    top: int = typer.Option(5, help="Number of results to return"),
) -> None:
    """Search memories for relevant past context."""
    from agent_memory.embedder import RuriEmbedder
    from agent_memory.search import hybrid_search
    from agent_memory.store import get_connection

    embedder = RuriEmbedder()
    query_vec = embedder.embed_query(query)

    conn = get_connection()
    project_str = str(project.resolve()) if project else None
    results = hybrid_search(conn, query, query_vec, project_str, top)
    conn.close()

    if not results:
        typer.echo("No results found.")
        return

    for i, r in enumerate(results, 1):
        typer.echo(f"\n[{i}] (score: {r.score:.4f}, {r.project_path}, {r.created_at})")
        typer.echo(r.chunk_text)


@app.command()
def summary() -> None:
    """Show summary of the memory database."""
    from agent_memory.store import get_connection

    conn = get_connection()

    row = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT session_id), COUNT(DISTINCT project_path), MIN(created_at), MAX(created_at) FROM memories"
    ).fetchone()
    total, sessions, projects, oldest, newest = row

    if total == 0:
        typer.echo("DB is empty.")
        conn.close()
        return

    vec_count = conn.execute("SELECT COUNT(*) FROM memories_vec").fetchone()[0]

    typer.echo(f"DB:       {DEFAULT_DB_PATH}")
    typer.echo(f"Chunks:   {total}")
    typer.echo(f"Vectors:  {vec_count}")
    typer.echo(f"Sessions: {sessions}")
    typer.echo(f"Projects: {projects}")
    typer.echo(f"Oldest:   {oldest}")
    typer.echo(f"Newest:   {newest}")

    typer.echo("\n--- Per project ---")
    rows = conn.execute(
        "SELECT project_path, COUNT(*), COUNT(DISTINCT session_id) FROM memories GROUP BY project_path ORDER BY COUNT(*) DESC"
    ).fetchall()
    for path, cnt, sess in rows:
        typer.echo(f"  {path}  chunks={cnt}  sessions={sess}")

    conn.close()


@app.command()
def debug() -> None:
    """Open the memory database in VS Code."""
    db_path = DEFAULT_DB_PATH
    if not db_path.exists():
        typer.echo(f"DB not found: {db_path}")
        raise typer.Exit(1)

    typer.echo(f"Opening {db_path}")
    subprocess.run(["code", str(db_path)])


if __name__ == "__main__":
    app()
