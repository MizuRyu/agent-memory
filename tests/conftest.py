from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from agent_memory.constants import DIMENSION
from agent_memory.store import get_connection


class FakeEmbedder:
    """Deterministic hash-based embedder for testing."""

    DIMENSION = DIMENSION

    def _text_to_vec(self, text: str) -> NDArray[np.float32]:
        # Use hash to seed a deterministic RNG for reproducible vectors
        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)
        vec = rng.randn(DIMENSION).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    def embed_documents(self, texts: list[str]) -> NDArray[np.float32]:
        return np.stack([self._text_to_vec(t) for t in texts])

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return self._text_to_vec(text)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def tmp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    conn = get_connection(db_path)
    yield conn, db_path
    conn.close()
    db_path.unlink(missing_ok=True)
