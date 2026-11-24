"""Vector stores.

* ``LocalStore``: NumPy matrix + JSON on disk, exact cosine search. Used by tests, the evals
  and the pre-commit hook, so they run without a database.
* ``PgVectorStore``: Postgres + pgvector, the deployment target (``docker compose up``).

Both implement the same ``VectorStore`` protocol and the same scoring (cosine similarity on
normalized vectors), so eval numbers carry over; pgvector's HNSW is approximate, but at this
corpus size its recall against exact search is effectively 1.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ingest.models import Chunk, Document, SearchHit


class IndexMismatchError(RuntimeError):
    pass


class VectorStore(Protocol):
    def write(
        self, docs: list[Document], chunks: list[Chunk], vectors: np.ndarray, meta: dict[str, Any]
    ) -> None: ...

    def meta(self) -> dict[str, Any]: ...

    def search(
        self, vector: np.ndarray, k: int, doc_type: str | None = None
    ) -> list[SearchHit]: ...

    def get_doc(self, doc_id: str) -> Document | None: ...

    def list_docs(self) -> list[Document]: ...


class LocalStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._docs: dict[str, Document] = {}
        self._chunks: list[Chunk] = []
        self._vectors = np.zeros((0, 0), dtype=np.float32)
        self._meta: dict[str, Any] = {}
        if path is not None and (path / "index.json").exists():
            self._load(path)

    def _load(self, path: Path) -> None:
        data = json.loads((path / "index.json").read_text(encoding="utf-8"))
        self._meta = data["meta"]
        self._docs = {d["doc_id"]: Document(**d) for d in data["docs"]}
        self._chunks = [Chunk(**c) for c in data["chunks"]]
        self._vectors = np.load(path / "vectors.npy")

    def write(
        self, docs: list[Document], chunks: list[Chunk], vectors: np.ndarray, meta: dict[str, Any]
    ) -> None:
        self._docs = {d.doc_id: d for d in docs}
        self._chunks = list(chunks)
        self._vectors = vectors.astype(np.float32)
        self._meta = dict(meta)
        if self.path is not None:
            self.path.mkdir(parents=True, exist_ok=True)
            payload = {
                "meta": self._meta,
                "docs": [d.model_dump() for d in docs],
                "chunks": [c.model_dump() for c in chunks],
            }
            (self.path / "index.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            np.save(self.path / "vectors.npy", self._vectors)

    def meta(self) -> dict[str, Any]:
        return dict(self._meta)

    def search(self, vector: np.ndarray, k: int, doc_type: str | None = None) -> list[SearchHit]:
        if not self._chunks:
            return []
        scores = self._vectors @ vector.astype(np.float32)
        order = np.argsort(-scores, kind="stable")
        hits: list[SearchHit] = []
        for i in order:
            chunk = self._chunks[int(i)]
            if doc_type and self._docs[chunk.doc_id].doc_type != doc_type:
                continue
            hits.append(SearchHit(chunk=chunk, score=float(scores[i])))
            if len(hits) >= k:
                break
        return hits

    def get_doc(self, doc_id: str) -> Document | None:
        return self._docs.get(doc_id)

    def list_docs(self) -> list[Document]:
        return sorted(self._docs.values(), key=lambda d: d.doc_id)


_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS kb_meta (key text PRIMARY KEY, value jsonb NOT NULL);
CREATE TABLE IF NOT EXISTS kb_docs (
  doc_id text PRIMARY KEY, title text NOT NULL, doc_type text NOT NULL, area text NOT NULL,
  version text NOT NULL, owner text NOT NULL, path text NOT NULL, body text NOT NULL
);
"""


class PgVectorStore:
    """Postgres + pgvector. ``dsn`` defaults to ``KB_DATABASE_URL``."""

    def __init__(self, dsn: str | None = None) -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        self._dsn = dsn or os.environ["KB_DATABASE_URL"]
        self._conn = psycopg.connect(self._dsn, autocommit=True)
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)

    def write(
        self, docs: list[Document], chunks: list[Chunk], vectors: np.ndarray, meta: dict[str, Any]
    ) -> None:
        dim = int(vectors.shape[1])
        with self._conn.transaction():
            self._conn.execute("DROP TABLE IF EXISTS kb_chunks")
            self._conn.execute("DROP TABLE IF EXISTS kb_docs")
            self._conn.execute("DROP TABLE IF EXISTS kb_meta")
            self._conn.execute(_SCHEMA)
            self._conn.execute(
                f"""CREATE TABLE kb_chunks (
                  chunk_id text PRIMARY KEY, doc_id text NOT NULL REFERENCES kb_docs(doc_id),
                  ordinal int NOT NULL, section text NOT NULL, start_off int NOT NULL,
                  end_off int NOT NULL, text text NOT NULL, embedding vector({dim}) NOT NULL)"""
            )
            with self._conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO kb_docs VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                    [
                        (d.doc_id, d.title, d.doc_type, d.area, d.version, d.owner, d.path, d.text)
                        for d in docs
                    ],
                )
                cur.executemany(
                    "INSERT INTO kb_chunks VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                    [
                        (c.chunk_id, c.doc_id, c.ordinal, c.section, c.start, c.end, c.text, v)
                        for c, v in zip(chunks, vectors, strict=True)
                    ],
                )
                cur.execute("INSERT INTO kb_meta VALUES ('index', %s::jsonb)", (json.dumps(meta),))
            self._conn.execute(
                "CREATE INDEX kb_chunks_hnsw ON kb_chunks USING hnsw (embedding vector_cosine_ops)"
            )

    def meta(self) -> dict[str, Any]:
        row = self._conn.execute("SELECT value FROM kb_meta WHERE key = 'index'").fetchone()
        return dict(row[0]) if row else {}

    def search(self, vector: np.ndarray, k: int, doc_type: str | None = None) -> list[SearchHit]:
        sql = """SELECT c.chunk_id, c.doc_id, c.ordinal, c.section, c.start_off, c.end_off,
                        c.text, 1 - (c.embedding <=> %(v)s) AS score
                 FROM kb_chunks c JOIN kb_docs d USING (doc_id)
                 WHERE (%(t)s::text IS NULL OR d.doc_type = %(t)s)
                 ORDER BY c.embedding <=> %(v)s LIMIT %(k)s"""
        rows = self._conn.execute(sql, {"v": vector, "t": doc_type, "k": k}).fetchall()
        return [
            SearchHit(
                chunk=Chunk(
                    chunk_id=r[0],
                    doc_id=r[1],
                    ordinal=r[2],
                    section=r[3],
                    start=r[4],
                    end=r[5],
                    text=r[6],
                ),
                score=float(r[7]),
            )
            for r in rows
        ]

    def _doc(self, row: tuple[Any, ...]) -> Document:
        keys = ("doc_id", "title", "doc_type", "area", "version", "owner", "path", "text")
        return Document(**dict(zip(keys, row, strict=True)))

    def get_doc(self, doc_id: str) -> Document | None:
        row = self._conn.execute("SELECT * FROM kb_docs WHERE doc_id = %s", (doc_id,)).fetchone()
        return self._doc(row) if row else None

    def list_docs(self) -> list[Document]:
        return [self._doc(r) for r in self._conn.execute("SELECT * FROM kb_docs ORDER BY doc_id")]


def get_store(kind: str | None = None) -> VectorStore:
    """``KB_STORE``: ``local`` (default, dir ``KB_INDEX_DIR``) or ``pgvector``."""
    kind = kind or os.getenv("KB_STORE", "local")
    if kind == "local":
        return LocalStore(Path(os.getenv("KB_INDEX_DIR", ".kb_index")))
    if kind == "pgvector":
        return PgVectorStore()
    raise ValueError(f"unknown store: {kind}")
