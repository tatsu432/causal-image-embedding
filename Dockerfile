# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONPATH=/app/src

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra dev --no-install-project

COPY src ./src
COPY tests ./tests

CMD ["uv", "run", "pytest", "tests/", "-q"]
