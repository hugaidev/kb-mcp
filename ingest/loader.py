"""Load Markdown documents with YAML front matter from a directory."""

from __future__ import annotations

from pathlib import Path

import yaml

from ingest.models import Document

REQUIRED = ("doc_id", "title", "doc_type", "area", "version")


class DocumentError(ValueError):
    pass


def parse_document(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    if not raw.startswith("---\n"):
        raise DocumentError(f"{path.name}: missing YAML front matter")
    _, fm, body = raw.split("---\n", 2)
    meta = yaml.safe_load(fm) or {}
    missing = [k for k in REQUIRED if not meta.get(k)]
    if missing:
        raise DocumentError(f"{path.name}: missing front matter fields {missing}")
    return Document(
        doc_id=str(meta["doc_id"]),
        title=str(meta["title"]),
        doc_type=str(meta["doc_type"]),
        area=str(meta["area"]),
        version=str(meta["version"]),
        owner=str(meta.get("owner", "")),
        path=path.as_posix(),
        text=body.strip("\n") + "\n",
    )


def load_documents(docs_dir: Path) -> list[Document]:
    docs = [parse_document(p) for p in sorted(docs_dir.glob("*.md"))]
    ids = [d.doc_id for d in docs]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise DocumentError(f"duplicate doc_id: {sorted(dupes)}")
    if not docs:
        raise DocumentError(f"no .md documents in {docs_dir}")
    return docs
