FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS uv

FROM python:3.12.12-slim

COPY --from=uv /usr/local/bin/uv /bin/uv

WORKDIR /app
ENV PYTHONPATH=/app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --group worker --no-install-project

COPY app ./app
COPY worker ./worker
RUN uv sync --frozen --no-dev --group worker

CMD ["uv", "run", "--group", "worker", "python", "-m", "worker.main"]
