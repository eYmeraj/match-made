FROM python:3.12-slim AS build

# sparse-dot-topn has no wheel for some platforms (e.g. linux/arm64) and builds from source.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.3 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Dependencies first so code changes don't reinstall them.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable


FROM python:3.12-slim

# OpenMP runtime used by sparse-dot-topn's multithreaded products.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /app/.venv /app/.venv

ENV PATH=/app/.venv/bin:$PATH \
    MATCHMADE_DATA_DIR=/data
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "matchmade.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
