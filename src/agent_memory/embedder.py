from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from agent_memory.constants import DEFAULT_MODEL_NAME, DOCUMENT_PREFIX, QUERY_PREFIX


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> NDArray[np.float32]: ...
    def embed_query(self, text: str) -> NDArray[np.float32]: ...


class RuriEmbedder:
    """Production embedder using cl-nagoya/ruri-v3-310m."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> NDArray[np.float32]:
        prefixed = [DOCUMENT_PREFIX + t for t in texts]
        return self._model.encode(prefixed, normalize_embeddings=True)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return self._model.encode(
            [QUERY_PREFIX + text], normalize_embeddings=True
        )[0]
