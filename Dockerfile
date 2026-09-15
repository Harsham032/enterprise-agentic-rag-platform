# Multi-stage build. The builder installs into a virtualenv which is copied into
# a clean runtime image, so compilers and build headers never reach production.
FROM python:3.11-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential libxml2-dev libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependencies are installed before the source is copied so that a code change
# does not invalidate the dependency layer.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-deps .


FROM python:3.11-slim-bookworm AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RAG_ENV=production

RUN apt-get update \
    && apt-get install --no-install-recommends -y libxml2 libxslt1.1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 rag

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=rag:rag configs ./configs
COPY --chown=rag:rag scripts ./scripts
COPY --chown=rag:rag data/fixtures ./data/fixtures
COPY --chown=rag:rag data/README.md ./data/README.md

RUN mkdir -p /app/data/processed /app/reports && chown -R rag:rag /app/data /app/reports

USER rag
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "rag_platform.services.api:app", "--host", "0.0.0.0", "--port", "8000"]
