"""``kb-ingest``: build the index from ``docs/``."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ingest.chunking import get_chunker
from ingest.embeddings import get_embedder
from ingest.pipeline import ingest
from ingest.store import get_store


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="kb-ingest", description=__doc__)
    p.add_argument("--docs", type=Path, default=Path(os.getenv("KB_DOCS_DIR", "docs")))
    p.add_argument(
        "--chunker", choices=["semantic", "fixed"], default=os.getenv("KB_CHUNKER", "semantic")
    )
    p.add_argument("--store", choices=["local", "pgvector"], default=os.getenv("KB_STORE", "local"))
    p.add_argument("--embedder", default=None, help="fastembed | fastembed:<model> | hash")
    args = p.parse_args(argv)

    embedder = get_embedder(args.embedder)
    result = ingest(args.docs, get_chunker(args.chunker, embedder), embedder, get_store(args.store))
    sizes = [len(c.text) for c in result.chunks]
    print(
        f"indexed {len(result.docs)} docs -> {len(result.chunks)} chunks "
        f"({args.chunker}, avg {sum(sizes) / len(sizes):.0f} chars) into {args.store} "
        f"with {embedder.name}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
