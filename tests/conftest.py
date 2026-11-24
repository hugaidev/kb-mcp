"""Tests use the hash embedder: offline, deterministic, fast. Retrieval quality is measured by
``kb-evals`` with the real model, never by these tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from ingest.chunking import SemanticChunker
from ingest.embeddings import HashEmbedder
from ingest.loader import load_documents
from ingest.models import Document
from ingest.pipeline import build
from ingest.store import LocalStore
from server.app import KnowledgeBase

DOCS = Path(__file__).resolve().parent.parent / "docs"


@pytest.fixture(scope="session")
def docs() -> list[Document]:
    return load_documents(DOCS)


@pytest.fixture(scope="session")
def embedder() -> HashEmbedder:
    return HashEmbedder()


@pytest.fixture
def kb(docs: list[Document], embedder: HashEmbedder, tmp_path: Path) -> KnowledgeBase:
    result = build(docs, SemanticChunker(embedder), embedder)
    store = LocalStore(tmp_path / "idx")
    store.write(
        result.docs,
        result.chunks,
        result.vectors,
        {"embedder": embedder.name, "chunker": "semantic"},
    )
    return KnowledgeBase(LocalStore(tmp_path / "idx"), embedder)
