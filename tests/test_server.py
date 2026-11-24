from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import Client

from ingest.embeddings import HashEmbedder
from ingest.store import IndexMismatchError, LocalStore
from server.app import KnowledgeBase, build_server


def test_search_docs_shape(kb: KnowledgeBase) -> None:
    out = kb.search_docs("¿cuántos días de vacaciones tengo?", k=3)
    assert len(out["results"]) == 3
    first = out["results"][0]
    assert set(first) >= {"rank", "doc_id", "title", "section", "chunk_id", "score", "text", "uri"}
    assert first["uri"] == f"kb://docs/{first['doc_id']}"


def test_search_docs_filters_and_clamps(kb: KnowledgeBase) -> None:
    out = kb.search_docs("gastos", k=500, doc_type="procedimiento")
    assert len(out["results"]) == 20
    assert all(r["doc_id"].startswith("PRC-") for r in out["results"])
    assert kb.search_docs("   ")["error"]


def test_get_doc(kb: KnowledgeBase) -> None:
    doc = kb.get_doc("POL-004")
    assert doc["title"] == "Política de licencias y vacaciones"
    assert "path" not in doc
    missing = kb.get_doc("POL-999")
    assert "unknown doc_id" in missing["error"]
    assert "POL-004" in missing["known_doc_ids"]


def test_refuses_empty_or_mismatched_index(tmp_path: Path) -> None:
    with pytest.raises(IndexMismatchError, match="empty"):
        KnowledgeBase(LocalStore(tmp_path / "none"), HashEmbedder())
    store = LocalStore()
    store.write([], [], HashEmbedder().embed([]), {"embedder": "fastembed:other"})
    with pytest.raises(IndexMismatchError, match="re-run"):
        KnowledgeBase(store, HashEmbedder())


def _with_client(kb: KnowledgeBase, fn: Any) -> Any:
    async def run() -> Any:
        async with Client(build_server(kb)) as client:
            return await fn(client)

    return anyio.run(run)


def test_mcp_lists_tools_and_resources(kb: KnowledgeBase) -> None:
    async def fn(client: Client) -> tuple[set[str], set[str], set[str]]:
        tools = {t.name for t in (await client.list_tools()).tools}
        resources = {str(r.uri) for r in (await client.list_resources()).resources}
        templates = {
            t.uri_template for t in (await client.list_resource_templates()).resource_templates
        }
        return tools, resources, templates

    tools, resources, templates = _with_client(kb, fn)
    assert tools == {"search_docs", "get_doc"}
    assert resources == {"kb://docs"}
    assert templates == {"kb://docs/{doc_id}"}


def test_mcp_call_search_and_read_resources(kb: KnowledgeBase) -> None:
    async def fn(client: Client) -> tuple[Any, ...]:
        res = await client.call_tool("search_docs", {"query": "contraseña", "k": 2})
        catalog = await client.read_resource("kb://docs")
        doc = await client.read_resource("kb://docs/PRC-003")
        return res, catalog, doc

    res, catalog, doc = _with_client(kb, fn)
    payload = res.structured_content or json.loads(res.content[0].text)
    assert len(payload["results"]) == 2
    items = json.loads(catalog.contents[0].text)
    assert len(items) == 10
    assert doc.contents[0].text.startswith("# Procedimiento de reporte de incidentes de seguridad")
