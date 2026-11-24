from __future__ import annotations

from pathlib import Path

import pytest

from ingest import cli
from ingest.loader import DocumentError, load_documents, parse_document
from ingest.store import LocalStore


def test_cli_builds_local_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KB_INDEX_DIR", str(tmp_path / "idx"))
    assert cli.main(["--embedder", "hash", "--chunker", "fixed", "--store", "local"]) == 0
    store = LocalStore(tmp_path / "idx")
    assert store.meta() == {"embedder": "hash:1024", "dim": 1024, "chunker": "fixed"}
    assert len(store.list_docs()) == 10


def test_loader_rejects_bad_documents(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# sin front matter\n", encoding="utf-8")
    with pytest.raises(DocumentError, match="front matter"):
        parse_document(tmp_path / "a.md")
    (tmp_path / "a.md").write_text("---\ndoc_id: X\n---\ncuerpo\n", encoding="utf-8")
    with pytest.raises(DocumentError, match="missing front matter fields"):
        parse_document(tmp_path / "a.md")
    with pytest.raises(DocumentError, match=r"no \.md"):
        load_documents(tmp_path / "empty")


def test_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    body = "---\ndoc_id: X\ntitle: T\ndoc_type: politica\narea: A\nversion: '1'\n---\nhola\n"
    (tmp_path / "a.md").write_text(body, encoding="utf-8")
    (tmp_path / "b.md").write_text(body, encoding="utf-8")
    with pytest.raises(DocumentError, match="duplicate"):
        load_documents(tmp_path)
