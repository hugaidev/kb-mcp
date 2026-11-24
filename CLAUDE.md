# kb-mcp

MCP server (Python, `mcp` SDK 2.x) that exposes a corporate knowledge base of policies and
procedures with RAG, plus retrieval evals. Python 3.12, uv, pydantic v2, fastembed, pgvector.

## Commands

- `python` is not on PATH on the dev machine: always go through `uv run`.
- `uv sync --all-extras`, then `uv run ruff format --check`, `uv run ruff check`, `uv run mypy`,
  `uv run pytest`. All four must pass before a commit.
- `uv run kb-ingest` builds `.kb_index/` (local store). `uv run kb-mcp` serves over stdio.
- `uv run kb-evals` runs the retrieval evals (fixed, fixed-360, semantic) and rewrites
  `evals/reports/latest.{md,json}`. `--gate` checks `evals/thresholds.yaml`.
- A PreToolUse hook runs `kb-evals --gate` before every `git commit` and blocks it on a
  regression (~45 s). Do not bypass it.

## Layout

- `docs/`: the corpus. One Markdown file per document with YAML front matter (`doc_id`, `title`,
  `doc_type` politica|procedimiento, `area`, `version`). Synthetic company "Nortia S.A.".
- `ingest/`: loader, chunkers, embedders, stores (LocalStore, PgVectorStore), pipeline, CLI.
- `server/app.py`: `KnowledgeBase` (plain logic, unit-tested) + `build_server` (MCP wiring).
- `evals/`: `golden.yaml` (40 cases), `harness.py`, `thresholds.yaml`, `reports/`.
- `.claude/`: hook `pre_commit_evals.py`, subagent `retrieval-reviewer`.

## Conventions

- Chunks are contiguous spans of the document body (`text == doc.text[start:end]`); the evals
  judge relevance by char overlap with the gold evidence, so never break that invariant.
- Both chunkers embed the same `title > section` prefix (`Chunk.embed_text`). Keep it shared so
  the fixed-vs-semantic delta measures boundaries only.
- Tests use `HashEmbedder` (offline, deterministic). It is a lexical baseline, not a semantic
  model: never quote its numbers as retrieval quality. Real numbers come from `kb-evals`.
- The stdio server must never print to stdout; log to stderr.
- MCP tool names, parameters and result keys are a client contract: change them deliberately.
- Reports stay deterministic: no timestamps.

## Rules

- Never edit, delete or reword golden cases, move an evidence span, or lower a floor in
  `thresholds.yaml` to turn a run green. Fix the regression, or change the floor in its own
  commit with the reason. New cases go at the end of `golden.yaml`.
- Evidence must be a verbatim substring of its document; questions must paraphrase, not copy.
- Synthetic data only. No secrets in the repo; `compose.yaml` credentials are local-only.
- Before committing changes to `ingest/`, `server/` or `evals/`, ask the `retrieval-reviewer`
  subagent for a review.
