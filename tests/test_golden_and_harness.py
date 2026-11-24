from __future__ import annotations

from pathlib import Path

import pytest

from evals import harness
from ingest.embeddings import HashEmbedder
from ingest.models import Chunk, Document


def test_golden_set_is_valid(docs: list[Document]) -> None:
    cases = harness.load_cases()
    assert len(cases) == 40
    assert len({c.id for c in cases}) == 40
    by_id = {d.doc_id: d for d in docs}
    for case in cases:
        assert case.doc_id in by_id, case.id
        # Raises if the evidence is not a verbatim substring of the gold document.
        harness.evidence_span(by_id[case.doc_id], case.evidence)
        # Questions must not copy the evidence wording.
        assert case.evidence.lower() not in case.question.lower(), case.id


def _chunk(start: int, end: int, doc_id: str = "D") -> Chunk:
    return Chunk(chunk_id="x", doc_id=doc_id, ordinal=0, section="", start=start, end=end, text="")


def test_relevance_by_coverage() -> None:
    span = (100, 200)
    assert harness.is_relevant(_chunk(90, 210), "D", span)
    assert harness.is_relevant(_chunk(120, 300), "D", span)  # covers 80 %
    assert not harness.is_relevant(_chunk(150, 300), "D", span)  # covers 50 %
    assert not harness.is_relevant(_chunk(90, 210, doc_id="E"), "D", span)


def test_metrics_math() -> None:
    case = harness.Case("q", "?", "D", "e")
    run = harness.RunResult(
        "fixed",
        1,
        1.0,
        [
            harness.CaseResult(case, 1, ["D"]),
            harness.CaseResult(case, 4, ["E", "E", "E", "D"]),
            harness.CaseResult(case, None, ["E"]),
            harness.CaseResult(case, 2, ["E", "D"]),
        ],
    )
    m = run.metrics()
    assert m["recall@1"] == 0.25
    assert m["recall@3"] == 0.5
    assert m["recall@5"] == 0.75
    assert m["mrr@10"] == round((1 + 0.25 + 0 + 0.5) / 4, 4)
    assert m["doc_hit@5"] == 0.75


def test_harness_end_to_end_with_hash(tmp_path: Path) -> None:
    rc = harness.main(
        ["--embedder", "hash", "--chunkers", "fixed,semantic", "--out", str(tmp_path)]
    )
    assert rc == 0
    md = (tmp_path / "latest.md").read_text(encoding="utf-8")
    assert "Δ semantic − fixed" in md
    assert (tmp_path / "latest.json").exists()


def test_gate_refuses_hash_embedder(tmp_path: Path) -> None:
    rc = harness.main(
        ["--embedder", "hash", "--chunkers", "semantic", "--out", str(tmp_path), "--gate"]
    )
    assert rc == 2


def test_check_gate() -> None:
    case = harness.Case("q", "?", "D", "e")
    run = harness.RunResult("semantic", 1, 1.0, [harness.CaseResult(case, None, [])])
    failures = harness.check_gate([run], {"semantic": {"recall@5": 0.5}, "other": {"mrr@10": 1}})
    assert failures == ["semantic recall@5 = 0.000 < 0.500"]


def test_evidence_not_found(docs: list[Document]) -> None:
    with pytest.raises(ValueError, match="not found verbatim"):
        harness.evidence_span(docs[0], "texto que no existe en ningún documento")


def test_hash_embedder_is_deterministic_and_normalized() -> None:
    e = HashEmbedder(dim=64)
    a, b = e.embed(["hola mundo", "hola mundo"])
    assert (a == b).all()
    assert abs(float((a * a).sum()) - 1.0) < 1e-5
