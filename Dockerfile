FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    FASTEMBED_CACHE_PATH=/opt/models \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, for layer caching.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra pg --no-install-project

COPY ingest ./ingest
COPY server ./server
COPY evals ./evals
COPY docs ./docs
RUN uv sync --frozen --no-dev --extra pg

# Bake the embedding model into the image: the container needs no network at runtime.
RUN uv run --no-sync python -c "from ingest.embeddings import FastEmbedder; FastEmbedder()"

RUN useradd --create-home --uid 10001 kb && chown -R kb /app /opt/models
USER kb

ENV KB_TRANSPORT=streamable-http KB_HOST=0.0.0.0 KB_PORT=8000 KB_DOCS_DIR=/app/docs
EXPOSE 8000

# Re-index on every start (seconds at this corpus size), then serve.
CMD ["sh", "-c", "uv run --no-sync kb-ingest && exec uv run --no-sync kb-mcp"]
