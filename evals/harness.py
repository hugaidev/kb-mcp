"""``kb-evals``: retrieval evals over the golden set, one run per chunker.

Relevance is chunker-independent: a retrieved chunk is relevant when it belongs to the gold
document and covers at least ``MIN_COVERAGE`` of the gold evidence span (char offsets).

Metrics per chunker:
* recall@k: share of questions with at least one relevant chunk in the top k.
* MRR@10: mean of 1/rank of the first relevant chunk (0 if none in the top 10).
* doc_hit@5: share of questions whose gold document appears in the top 5 (lenient).

Reports are deterministic (no timestamps) so they can be diffed in PRs.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ingest.chunking import get_chunker
from ingest.embeddings import Embedder, get_embedder
from ingest.loader import load_documents
from ingest.models import Chunk, Document
from ingest.pipeline import build
from ingest.store import LocalStore

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "evals" / "golden.yaml"
THRESHOLDS = ROOT / "evals" / "thresholds.yaml"
REPORTS = ROOT / "evals" / "reports"
MIN_COVERAGE = 0.8
KS = (1, 3, 5)
MRR_DEPTH = 10


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    doc_id: str
    evidence: str


@dataclass
class CaseResult:
    case: Case
    first_relevant_rank: int | None
    top_doc_ids: list[str]


@dataclass
class RunResult:
    chunker: str
    n_chunks: int
    avg_chunk_chars: float
    cases: list[CaseResult] = field(default_factory=list)

    def recall_at(self, k: int) -> float:
        hits = sum(1 for c in self.cases if c.first_relevant_rank and c.first_relevant_rank <= k)
        return hits / len(self.cases)

    def mrr(self) -> float:
        return sum(1 / c.first_relevant_rank for c in self.cases if c.first_relevant_rank) / len(
            self.cases
        )

    def doc_hit_at(self, k: int) -> float:
        return sum(1 for c in self.cases if c.case.doc_id in c.top_doc_ids[:k]) / len(self.cases)

    def metrics(self) -> dict[str, float]:
        out = {f"recall@{k}": round(self.recall_at(k), 4) for k in KS}
        out["mrr@10"] = round(self.mrr(), 4)
        out["doc_hit@5"] = round(self.doc_hit_at(5), 4)
        return out


def load_cases(path: Path = GOLDEN) -> list[Case]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [Case(**c) for c in data["cases"]]


def evidence_span(doc: Document, evidence: str) -> tuple[int, int]:
    idx = doc.text.find(evidence)
    if idx < 0:
        raise ValueError(f"evidence not found verbatim in {doc.doc_id}: {evidence!r}")
    return idx, idx + len(evidence)


def is_relevant(chunk: Chunk, doc_id: str, span: tuple[int, int]) -> bool:
    if chunk.doc_id != doc_id:
        return False
    overlap = max(0, min(chunk.end, span[1]) - max(chunk.start, span[0]))
    return overlap / (span[1] - span[0]) >= MIN_COVERAGE


def run(
    chunker_name: str, docs: list[Document], cases: list[Case], embedder: Embedder
) -> RunResult:
    result = build(docs, get_chunker(chunker_name, embedder), embedder)
    store = LocalStore()
    store.write(result.docs, result.chunks, result.vectors, {"embedder": embedder.name})
    by_id = {d.doc_id: d for d in docs}
    qvecs = embedder.embed([c.question for c in cases])
    sizes = [len(c.text) for c in result.chunks]
    out = RunResult(chunker_name, len(result.chunks), sum(sizes) / len(sizes))
    for case, qv in zip(cases, qvecs, strict=True):
        span = evidence_span(by_id[case.doc_id], case.evidence)
        hits = store.search(qv, MRR_DEPTH)
        rank = next(
            (i + 1 for i, h in enumerate(hits) if is_relevant(h.chunk, case.doc_id, span)), None
        )
        out.cases.append(CaseResult(case, rank, [h.chunk.doc_id for h in hits]))
    return out


def render_markdown(runs: list[RunResult], embedder_name: str) -> str:
    names = [r.chunker for r in runs]
    metrics = {r.chunker: r.metrics() for r in runs}
    delta = "fixed" in metrics and "semantic" in metrics
    keys = list(next(iter(metrics.values())).keys())
    head = "| Metric | " + " | ".join(names) + " |" + (" Δ semantic − fixed |" if delta else "")
    lines = [
        "# Retrieval eval report",
        "",
        f"- Embedder: `{embedder_name}`",
        f"- Questions: {len(runs[0].cases)} · relevance: chunk covers >= {MIN_COVERAGE:.0%} "
        "of the gold evidence span",
        "",
        head,
        "|---|" + "---:|" * (len(names) + int(delta)),
    ]
    for key in keys:
        row = f"| {key} | " + " | ".join(f"{metrics[n][key]:.3f}" for n in names) + " |"
        if delta:
            row += f" {metrics['semantic'][key] - metrics['fixed'][key]:+.3f} |"
        lines.append(row)
    pad = " |" if delta else ""
    lines.append("| chunks | " + " | ".join(str(r.n_chunks) for r in runs) + " |" + pad)
    lines.append(
        "| avg chunk chars | " + " | ".join(f"{r.avg_chunk_chars:.0f}" for r in runs) + " |" + pad
    )
    if delta:
        fixed = {
            c.case.id: c.first_relevant_rank or 99
            for r in runs
            if r.chunker == "fixed"
            for c in r.cases
        }
        sem = {
            c.case.id: c.first_relevant_rank or 99
            for r in runs
            if r.chunker == "semantic"
            for c in r.cases
        }
        wins = sum(1 for q in sem if sem[q] < fixed[q])
        losses = sum(1 for q in sem if sem[q] > fixed[q])
        lines += [
            "",
            f"Paired by question (rank of first relevant chunk): semantic better in {wins}, "
            f"worse in {losses}, tied in {len(sem) - wins - losses}.",
        ]
    lines += ["", "## Misses (no relevant chunk in top 5)", ""]
    for r in runs:
        misses = [c for c in r.cases if not c.first_relevant_rank or c.first_relevant_rank > 5]
        lines.append(f"**{r.chunker}** ({len(misses)})")
        lines.append("")
        for c in misses:
            rank = c.first_relevant_rank or "—"
            lines.append(
                f"- `{c.case.id}` {c.case.question} → gold `{c.case.doc_id}`, first relevant "
                f"rank {rank}, top-3 docs {c.top_doc_ids[:3]}"
            )
        lines.append("")
    return "\n".join(lines)


def check_gate(runs: list[RunResult], thresholds: dict[str, dict[str, float]]) -> list[str]:
    failures = []
    by_name = {r.chunker: r.metrics() for r in runs}
    for chunker, mins in thresholds.items():
        if chunker not in by_name:
            continue
        for metric, minimum in mins.items():
            value = by_name[chunker][metric]
            if value < minimum:
                failures.append(f"{chunker} {metric} = {value:.3f} < {minimum:.3f}")
    return failures


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="kb-evals", description="Retrieval evals (recall@k, MRR).")
    p.add_argument("--chunkers", default="fixed,fixed-360,semantic")
    p.add_argument("--embedder", default=None, help="fastembed | fastembed:<model> | hash")
    p.add_argument("--docs", type=Path, default=ROOT / "docs")
    p.add_argument("--out", type=Path, default=REPORTS)
    p.add_argument("--gate", action="store_true", help="exit 1 if below evals/thresholds.yaml")
    args = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    embedder = get_embedder(args.embedder)
    docs = load_documents(args.docs)
    cases = load_cases()
    runs = [run(name, docs, cases, embedder) for name in args.chunkers.split(",")]

    args.out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "embedder": embedder.name,
        "min_coverage": MIN_COVERAGE,
        "runs": {
            r.chunker: {
                "metrics": r.metrics(),
                "n_chunks": r.n_chunks,
                "avg_chunk_chars": round(r.avg_chunk_chars, 1),
                "first_relevant_rank": {c.case.id: c.first_relevant_rank for c in r.cases},
            }
            for r in runs
        },
    }
    (args.out / "latest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = render_markdown(runs, embedder.name)
    (args.out / "latest.md").write_text(md + "\n", encoding="utf-8")
    print(md.split("## Misses")[0].rstrip())

    if args.gate:
        if embedder.name.startswith("hash"):
            print("gate: refusing to gate on the hash embedder (test-only)", file=sys.stderr)
            return 2
        thresholds = yaml.safe_load(THRESHOLDS.read_text(encoding="utf-8"))["min"]
        failures = check_gate(runs, thresholds)
        if failures:
            print("EVAL GATE FAILED:\n  " + "\n  ".join(failures), file=sys.stderr)
            return 1
        print("eval gate passed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
