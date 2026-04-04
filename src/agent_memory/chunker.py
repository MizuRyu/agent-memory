from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agent_memory.constants import MAX_CHUNK_CHARS


@dataclass
class Chunk:
    chunk_index: int
    chunk_text: str


@dataclass
class ChunkResult:
    session_id: str
    chunks: list[Chunk]


def _extract_text(message: dict) -> str:
    """Extract text content from a transcript message."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts).strip()
    return ""


def _split_long_chunk(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split a long chunk at paragraph boundaries."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split("\n\n")
    result: list[str] = []
    current = ""

    for para in paragraphs:
        if current and len(current) + len(para) + 2 > max_chars:
            result.append(current.strip())
            current = para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        result.append(current.strip())

    return result if result else [text]


def chunk_transcript(transcript_path: Path) -> ChunkResult:
    """Parse a Claude Code JSONL transcript into Q&A chunks."""
    lines = transcript_path.read_text(encoding="utf-8").strip().splitlines()

    session_id = ""
    turns: list[tuple[str, str]] = []  # (role, text)

    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        entry_type = entry.get("type")
        if entry_type not in ("user", "assistant"):
            continue

        # Skip meta messages (skill injections, system messages)
        if entry.get("isMeta", False):
            continue

        if not session_id:
            session_id = entry.get("sessionId", "")

        message = entry.get("message", {})
        text = _extract_text(message)
        if not text:
            continue

        role = message.get("role", entry_type)
        turns.append((role, text))

    # Group into user-assistant pairs
    chunks: list[Chunk] = []
    chunk_index = 0
    i = 0

    while i < len(turns):
        # Collect consecutive user turns
        user_parts: list[str] = []
        while i < len(turns) and turns[i][0] == "user":
            user_parts.append(turns[i][1])
            i += 1

        # Collect consecutive assistant turns
        assistant_parts: list[str] = []
        while i < len(turns) and turns[i][0] == "assistant":
            assistant_parts.append(turns[i][1])
            i += 1

        if not user_parts and not assistant_parts:
            i += 1
            continue

        user_text = "\n".join(user_parts)
        assistant_text = "\n".join(assistant_parts)

        if user_text and assistant_text:
            chunk_text = f"User: {user_text}\n\nAssistant: {assistant_text}"
        elif user_text:
            chunk_text = f"User: {user_text}"
        else:
            chunk_text = f"Assistant: {assistant_text}"

        for part in _split_long_chunk(chunk_text):
            chunks.append(Chunk(chunk_index=chunk_index, chunk_text=part))
            chunk_index += 1

    return ChunkResult(session_id=session_id or "unknown", chunks=chunks)
