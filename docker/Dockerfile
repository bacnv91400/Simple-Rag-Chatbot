FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS uv

FROM python:3.12.12-slim

COPY --from=uv /usr/local/bin/uv /bin/uv

# libmagic1 is required at runtime by python-magic (Milestone 2 format check)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libmagic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
RUN uv sync --frozen --no-dev

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app/main.py", "--server.address=0.0.0.0", "--server.port=8501"]