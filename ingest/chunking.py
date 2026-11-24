"""Two chunking strategies over the same document body.

* ``fixed``: character windows with overlap, snapped to whitespace. Ignores structure.
* ``semantic``: never crosses a Markdown heading; inside a section it splits where the
  embedding distance between consecutive sentences spikes, within size bounds.

Every chunk is a contiguous span ``text[start:end]`` of the body, so the evals can judge
relevance by character overlap with a gold evidence span, whatever the chunker.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ingest.embeddings import Embedder
from ingest.models import Chunk, Document

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
# Sentence boundary: end punctuation, whitespace, then something that starts a sentence.
_SENT_END = re.compile(r"(?<=[.!?:])\s+(?=[A-ZÁÉÍÓÚÑ¿¡0-9\"(])")


@dataclass(frozen=True)
class Unit:
    start: int
    end: int
    is_heading: bool
    section: str


def _sections(text: str) -> list[tuple[int, str]]:
    """(offset, heading text) for each heading of level >= 2."""
    out: list[tuple[int, str]] = []
    pos = 0
    for line in text.split("\n"):
        m = _HEADING.match(line)
        if m and len(m.group(1)) >= 2:
            out.append((pos, m.group(2)))
        pos += len(line) + 1
    return out


def section_at(sections: list[tuple[int, str]], offset: int) -> str:
    current = ""
    for start, name in sections:
        if start > offset:
            break
        current = name
    return current


def split_units(text: str) -> list[Unit]:
    """Headings and sentences with their offsets. List items and table rows are one unit."""
    units: list[Unit] = []
    section = ""
    pos = 0
    for line in text.split("\n"):
        line_start = pos
        pos += len(line) + 1
        if not line.strip():
            continue
        m = _HEADING.match(line)
        if m:
            if len(m.group(1)) >= 2:
                section = m.group(2)
            units.append(Unit(line_start, line_start + len(line), True, section))
            continue
        cursor = 0
        for piece in _SENT_END.split(line):
            idx = line.index(piece, cursor)
            cursor = idx + len(piece)
            if piece.strip():
                units.append(Unit(line_start + idx, line_start + cursor, False, section))
    return units


class Chunker(Protocol):
    name: str

    def chunk(self, doc: Document) -> list[Chunk]: ...


def _make(doc: Document, chunker: str, ordinal: int, start: int, end: int, section: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}#{chunker}-{ordinal:03d}",
        doc_id=doc.doc_id,
        ordinal=ordinal,
        section=section,
        start=start,
        end=end,
        text=doc.text[start:end],
    )


class FixedChunker:
    def __init__(self, size: int = 700, overlap: int = 100, name: str = "fixed") -> None:
        if not 0 <= overlap < size:
            raise ValueError("overlap must be in [0, size)")
        self.size = size
        self.overlap = overlap
        self.name = name

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text
        n = len(text.rstrip())
        sections = _sections(text)
        chunks: list[Chunk] = []
        start = 0
        while start < n:
            end = min(start + self.size, n)
            if end < n:
                cut = text.rfind(" ", start + self.size // 2, end)
                cut = max(cut, text.rfind("\n", start + self.size // 2, end))
                end = cut if cut > start else end
            s, e = start, end
            while s < e and text[s].isspace():
                s += 1
            while e > s and text[e - 1].isspace():
                e -= 1
            if e > s:
                chunks.append(_make(doc, self.name, len(chunks), s, e, section_at(sections, s)))
            if end >= n:
                break
            nxt = max(end - self.overlap, start + 1)
            while nxt < end and not text[nxt - 1].isspace():
                nxt += 1
            start = nxt
        return chunks


class SemanticChunker:
    def __init__(
        self,
        embedder: Embedder,
        max_chars: int = 900,
        min_chars: int = 200,
        breakpoint_percentile: float = 80.0,
    ) -> None:
        self.embedder = embedder
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.pct = breakpoint_percentile
        self.name = "semantic"

    def _groups_by_section(self, units: list[Unit]) -> list[list[Unit]]:
        groups: list[list[Unit]] = []
        for u in units:
            if u.is_heading or not groups:
                groups.append([u])
            else:
                groups[-1].append(u)
        return groups

    def chunk(self, doc: Document) -> list[Chunk]:
        units = split_units(doc.text)
        if not units:
            return []
        body = [u for u in units if not u.is_heading]
        vecs = self.embedder.embed([doc.text[u.start : u.end] for u in body])
        vec_of = {u: vecs[i] for i, u in enumerate(body)}
        dists = [
            1.0 - float(np.dot(vec_of[a], vec_of[b]))
            for a, b in itertools.pairwise(body)
            if a.section == b.section
        ]
        threshold = float(np.percentile(dists, self.pct)) if dists else 1.0

        spans: list[tuple[int, int, str]] = []
        for group in self._groups_by_section(units):
            cur_start = group[0].start
            cur_end = group[0].end
            prev: Unit | None = group[0] if not group[0].is_heading else None
            for u in group[1:]:
                size = cur_end - cur_start
                semantic_break = (
                    prev is not None
                    and 1.0 - float(np.dot(vec_of[prev], vec_of[u])) > threshold
                    and size >= self.min_chars
                )
                if semantic_break or (u.end - cur_start) > self.max_chars:
                    spans.append((cur_start, cur_end, group[0].section))
                    cur_start = u.start
                cur_end = u.end
                prev = u
            spans.append((cur_start, cur_end, group[0].section))

        merged: list[tuple[int, int, str]] = []
        for s, e, sec in spans:
            if (
                merged
                and merged[-1][2] == sec
                and (e - s < self.min_chars or merged[-1][1] - merged[-1][0] < self.min_chars)
                and e - merged[-1][0] <= int(self.max_chars * 1.3)
            ):
                merged[-1] = (merged[-1][0], e, sec)
            else:
                merged.append((s, e, sec))
        return [_make(doc, self.name, i, s, e, sec) for i, (s, e, sec) in enumerate(merged)]


def get_chunker(name: str, embedder: Embedder) -> Chunker:
    if name == "fixed":
        return FixedChunker()
    if name.startswith("fixed-"):
        # fixed-<size>: same algorithm, other window; overlap stays at ~15 %.
        size = int(name.split("-", 1)[1])
        return FixedChunker(size=size, overlap=size * 15 // 100, name=name)
    if name == "semantic":
        return SemanticChunker(embedder)
    raise ValueError(f"unknown chunker: {name}")
