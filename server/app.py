"""kb-mcp: MCP server over the corporate knowledge base.

Tools: ``search_docs`` (semantic search over chunks) and ``get_doc`` (full document).
Resources: ``kb://docs`` (catalog) and ``kb://docs/{doc_id}`` (one document as Markdown).

Transport is ``stdio`` by default (Claude Desktop / Claude Code launch it as a subprocess);
``KB_TRANSPORT=streamable-http`` serves it over HTTP, which is what the Docker image does.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations

from ingest.embeddings import Embedder, get_embedder
from ingest.store import IndexMismatchError, VectorStore, get_store

INSTRUCTIONS = (
    "Internal policies and procedures knowledge base. Call search_docs with the employee's "
    "question in natural language, then cite doc_id and section in the answer. Use get_doc "
    "only when the excerpts are not enough. If no result is relevant, say the policy was not "
    "found instead of guessing."
)

MAX_K = 20

# Both tools only read the local index: no side effects, same answer for the same input while
# the index is unchanged, and no external systems involved. Without these hints the protocol
# defaults assume a tool is destructive and open-world.
READ_ONLY = {
    "read_only_hint": True,
    "destructive_hint": False,
    "idempotent_hint": True,
    "open_world_hint": False,
}


class KnowledgeBase:
    """The tool logic, kept free of MCP types so it is unit-testable."""

    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        meta = store.meta()
        if not meta:
            raise IndexMismatchError("index is empty: run `kb-ingest` first")
        if meta.get("embedder") != embedder.name:
            raise IndexMismatchError(
                f"index built with {meta.get('embedder')}, server uses {embedder.name}; "
                "re-run `kb-ingest`"
            )
        self.store = store
        self.embedder = embedder

    def search_docs(self, query: str, k: int = 5, doc_type: str | None = None) -> dict[str, Any]:
        query = query.strip()
        if not query:
            return {"error": "query must not be empty", "results": []}
        k = max(1, min(int(k), MAX_K))
        vec = self.embedder.embed([query])[0]
        hits = self.store.search(vec, k, doc_type or None)
        titles = {d.doc_id: d.title for d in self.store.list_docs()}
        return {
            "query": query,
            "results": [
                {
                    "rank": i + 1,
                    "doc_id": h.chunk.doc_id,
                    "title": titles.get(h.chunk.doc_id, ""),
                    "section": h.chunk.section,
                    "chunk_id": h.chunk.chunk_id,
                    "score": round(h.score, 4),
                    "text": h.chunk.text,
                    "uri": f"kb://docs/{h.chunk.doc_id}",
                }
                for i, h in enumerate(hits)
            ],
        }

    def get_doc(self, doc_id: str) -> dict[str, Any]:
        doc = self.store.get_doc(doc_id.strip())
        if doc is None:
            known = [d.doc_id for d in self.store.list_docs()]
            return {"error": f"unknown doc_id: {doc_id}", "known_doc_ids": known}
        return doc.model_dump(exclude={"path"})

    def catalog(self) -> list[dict[str, str]]:
        return [
            {
                "doc_id": d.doc_id,
                "title": d.title,
                "doc_type": d.doc_type,
                "area": d.area,
                "version": d.version,
                "uri": f"kb://docs/{d.doc_id}",
            }
            for d in self.store.list_docs()
        ]


def build_server(kb: KnowledgeBase) -> MCPServer:
    mcp = MCPServer("kb-mcp", instructions=INSTRUCTIONS)

    @mcp.tool(
        title="Search policies",
        annotations=ToolAnnotations(title="Search policies", **READ_ONLY),
    )
    def search_docs(query: str, k: int = 5, doc_type: str | None = None) -> dict[str, Any]:
        """Search internal policies and procedures. Returns the k most relevant excerpts.

        query: the question in natural language (Spanish or English).
        k: number of excerpts, 1-20.
        doc_type: optional filter, "politica" or "procedimiento".
        """
        return kb.search_docs(query, k, doc_type)

    @mcp.tool(
        title="Get document",
        annotations=ToolAnnotations(title="Get document", **READ_ONLY),
    )
    def get_doc(doc_id: str) -> dict[str, Any]:
        """Return a full document (metadata and Markdown body) by doc_id, e.g. POL-001."""
        return kb.get_doc(doc_id)

    @mcp.resource(
        "kb://docs",
        name="catalog",
        mime_type="application/json",
        description="Catalog of every document in the knowledge base.",
    )
    def catalog() -> str:
        return json.dumps(kb.catalog(), ensure_ascii=False, indent=2)

    @mcp.resource(
        "kb://docs/{doc_id}",
        name="document",
        mime_type="text/markdown",
        description="One document as Markdown.",
    )
    def document(doc_id: str) -> str:
        doc = kb.store.get_doc(doc_id)
        if doc is None:
            raise ValueError(f"unknown doc_id: {doc_id}")
        return f"# {doc.title}\n\n_{doc.doc_id} · v{doc.version} · {doc.area}_\n\n{doc.text}"

    return mcp


def main() -> None:
    # stdout belongs to the stdio transport: logs go to stderr.
    logging.basicConfig(stream=sys.stderr, level=os.getenv("KB_LOG_LEVEL", "WARNING"))
    kb = KnowledgeBase(get_store(), get_embedder())
    server = build_server(kb)
    transport = os.getenv("KB_TRANSPORT", "stdio")
    if transport == "streamable-http":
        server.run(
            "streamable-http",
            host=os.getenv("KB_HOST", "127.0.0.1"),
            port=int(os.getenv("KB_PORT", "8000")),
        )
    else:
        server.run()


if __name__ == "__main__":
    main()
