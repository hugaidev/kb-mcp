"""Embedding backends. All return L2-normalized float32 matrices."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Protocol

import numpy as np

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out: np.ndarray = (m / norms).astype(np.float32)
    return out


class FastEmbedder:
    """Local ONNX model via fastembed. Multilingual because the corpus is in Spanish.

    The model is downloaded on first use and cached (``FASTEMBED_CACHE_PATH``).
    No API keys and no data leaves the machine.
    """

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(
            model_name=model,
            cache_dir=os.getenv("FASTEMBED_CACHE_PATH", str(Path.home() / ".cache" / "fastembed")),
        )
        self.name = f"fastembed:{model}"
        self.dim = int(self._model.embedding_size)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.array(list(self._model.embed(texts, batch_size=32))))


_TOKEN = re.compile(r"\w+", re.UNICODE)


class HashEmbedder:
    """Deterministic hashed bag of words + char trigrams. Offline, for tests only.

    It is a lexical baseline, not a semantic model: never report its numbers as RAG quality.
    """

    def __init__(self, dim: int = 1024) -> None:
        self.name = f"hash:{dim}"
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        words = [w.lower() for w in _TOKEN.findall(text)]
        grams = [w[i : i + 3] for w in words if len(w) > 3 for i in range(len(w) - 2)]
        return words + grams

    def embed(self, texts: list[str]) -> np.ndarray:
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat in self._features(text):
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
                m[row, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return _normalize(m)


def get_embedder(spec: str | None = None) -> Embedder:
    """``KB_EMBEDDER``: ``fastembed`` (default), ``fastembed:<model>`` or ``hash``."""
    spec = spec or os.getenv("KB_EMBEDDER") or "fastembed"
    if spec == "hash":
        return HashEmbedder()
    if spec == "fastembed":
        return FastEmbedder()
    if spec.startswith("fastembed:"):
        return FastEmbedder(spec.split(":", 1)[1])
    raise ValueError(f"unknown embedder: {spec}")
