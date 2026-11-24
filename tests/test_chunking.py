from __future__ import annotations

import itertools

import pytest

from ingest.chunking import FixedChunker, SemanticChunker, get_chunker, split_units
from ingest.embeddings import HashEmbedder
from ingest.models import Document


def _assert_spans(doc: Document, chunks: list) -> None:
    assert chunks
    for c in chunks:
        assert c.text == doc.text[c.start : c.end]
        assert c.text == c.text.strip()
        assert c.doc_id == doc.doc_id
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_fixed_spans_cover_document(docs: list[Document]) -> None:
    chunker = FixedChunker(size=300, overlap=50)
    for doc in docs:
        chunks = chunker.chunk(doc)
        _assert_spans(doc, chunks)
        assert all(len(c.text) <= 300 for c in chunks)
        # Consecutive windows overlap or touch: no text is dropped.
        for a, b in itertools.pairwise(chunks):
            assert b.start <= a.end + 1
        assert chunks[-1].end == len(doc.text.rstrip())


def test_semantic_never_crosses_headings(docs: list[Document], embedder: HashEmbedder) -> None:
    chunker = SemanticChunker(embedder, max_chars=600, min_chars=150)
    for doc in docs:
        chunks = chunker.chunk(doc)
        _assert_spans(doc, chunks)
        for c in chunks:
            # A heading may only open a chunk, never appear in the middle of one.
            assert "\n#" not in c.text


def test_semantic_respects_max_size(docs: list[Document], embedder: HashEmbedder) -> None:
    chunker = SemanticChunker(embedder, max_chars=600, min_chars=150)
    for doc in docs:
        for c in chunker.chunk(doc):
            assert len(c.text) <= int(600 * 1.3)


def test_split_units_offsets() -> None:
    text = "## Uno\n\nPrimera oración. Segunda oración.\n- item a\n"
    units = split_units(text)
    pieces = [text[u.start : u.end] for u in units]
    assert pieces == ["## Uno", "Primera oración.", "Segunda oración.", "- item a"]
    assert units[0].is_heading
    assert all(u.section == "Uno" for u in units)


def test_get_chunker_variants(embedder: HashEmbedder) -> None:
    assert get_chunker("fixed", embedder).name == "fixed"
    small = get_chunker("fixed-400", embedder)
    assert isinstance(small, FixedChunker) and small.size == 400
    assert get_chunker("semantic", embedder).name == "semantic"
    with pytest.raises(ValueError, match="unknown chunker"):
        get_chunker("nope", embedder)
    with pytest.raises(ValueError, match="overlap"):
        FixedChunker(size=100, overlap=100)
