"""Domain models shared by ingestion, the MCP server and the evals."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A source document (policy or procedure) loaded from ``docs/``."""

    doc_id: str
    title: str
    doc_type: str = Field(description="politica | procedimiento")
    area: str
    version: str
    owner: str = ""
    path: str
    text: str = Field(description="Markdown body without the front matter.")


class Chunk(BaseModel):
    """A contiguous span of a document's body. ``start``/``end`` are char offsets into it."""

    chunk_id: str
    doc_id: str
    ordinal: int
    section: str
    start: int
    end: int
    text: str

    def embed_text(self, title: str) -> str:
        """Text that gets embedded: title and section give the chunk its context.

        Both chunkers use the same prefix so the comparison isolates where the
        boundaries fall, not how much context each chunk carries.
        """
        header = f"{title} > {self.section}" if self.section else title
        return f"{header}\n{self.text}"


class SearchHit(BaseModel):
    chunk: Chunk
    score: float
