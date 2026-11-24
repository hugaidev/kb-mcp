"""Ingestion: load -> chunk -> embed -> write to a store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ingest.chunking import Chunker
from ingest.embeddings import Embedder
from ingest.loader import load_documents
from ingest.models import Chunk, Document
from ingest.store import VectorStore


@dataclass
class IngestResult:
    docs: list[Document]
    chunks: list[Chunk]
    vectors: np.ndarray


def build(docs: list[Document], chunker: Chunker, embedder: Embedder) -> IngestResult:
    chunks = [c for d in docs for c in chunker.chunk(d)]
    titles = {d.doc_id: d.title for d in docs}
    vectors = embedder.embed([c.embed_text(titles[c.doc_id]) for c in chunks])
    return IngestResult(docs, chunks, vectors)


def ingest(
    docs_dir: Path, chunker: Chunker, embedder: Embedder, store: VectorStore
) -> IngestResult:
    result = build(load_documents(docs_dir), chunker, embedder)
    store.write(
        result.docs,
        result.chunks,
        result.vectors,
        {"embedder": embedder.name, "dim": embedder.dim, "chunker": chunker.name},
    )
    return result
