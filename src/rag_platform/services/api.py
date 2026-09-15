"""FastAPI application.

The index is built once at startup and held in process. That is the right shape
for the default embedding backend, which is fit on the corpus: query vectors and
document vectors must come from the same fitted projection, so they have to live
together. Deployments using a pretrained backend can move the vectors into
pgvector or Qdrant and serve several replicas from shared storage.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from .. import __version__
from ..agents.orchestrator import Orchestrator
from ..config import PipelineConfig, Settings, load_settings
from ..data.local_documents import load_directory
from ..data.models import Document, DocumentMetadata
from ..data.store import MetadataStore, redact_url
from ..errors import EvaluationError, IndexNotBuiltError, RagPlatformError
from ..evaluation.harness import evaluate_retrieval
from ..evaluation.qrels import load_qrels
from ..logging_utils import configure_logging, get_logger
from ..retrieval.index import RetrievalIndex
from ..utils.seeds import set_global_seed
from ..utils.timing import Stopwatch
from .schemas import (
    AnswerResponse,
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
)

logger = get_logger(__name__)

QUERY_COUNTER = Counter(
    "rag_queries_total", "Queries served, labelled by planning strategy", ["strategy"]
)
QUERY_LATENCY = Histogram(
    "rag_query_latency_seconds",
    "End-to-end query latency",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
GROUNDEDNESS = Histogram(
    "rag_answer_groundedness",
    "Fraction of answer sentences supported by retrieved evidence",
    buckets=(0.0, 0.25, 0.5, 0.75, 0.9, 1.0),
)
INGESTED = Counter("rag_documents_ingested_total", "Documents added through the API")


class ServiceState:
    """Objects shared by every request."""

    def __init__(self) -> None:
        self.settings: Settings | None = None
        self.config: PipelineConfig | None = None
        self.index: RetrievalIndex | None = None
        self.orchestrator: Orchestrator | None = None
        self.store: MetadataStore | None = None
        self.error: str | None = None


state = ServiceState()


def build_state(config_path: str = "configs/default.yaml") -> ServiceState:
    """Load configuration, build the index and wire up persistence."""
    settings = load_settings()
    configure_logging(settings.log_level, json_output=settings.is_production)
    config = PipelineConfig.from_yaml(config_path)
    set_global_seed(config.run.seed)

    state.settings = settings
    state.config = config
    try:
        documents = load_directory(Path(config.corpus.source_dir))
        index = RetrievalIndex(config)
        index.build_from_documents(documents)
        # A reranker fitted offline by scripts/build_index.py; without it the
        # serving path falls back to the first-stage fused score.
        index.load_reranker(Path(config.run.output_dir) / "reranker.joblib")

        store = MetadataStore(settings.database_url)
        store.create_all()
        store.upsert_documents(index.documents)
        store.upsert_chunks(index.chunks)

        state.index = index
        state.store = store
        state.orchestrator = Orchestrator(index, config)
        state.error = None
        logger.info("service_ready", **index.describe())
    except RagPlatformError as exc:
        # A service that cannot build its index should still answer /health so
        # an operator can see why, rather than failing to start at all.
        state.error = str(exc)
        logger.error("service_startup_failed", error=str(exc))
    return state


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    build_state()
    yield
    if state.store is not None:
        state.store.close()


app = FastAPI(
    title="Retrieval and knowledge platform",
    description=(
        "Hybrid retrieval over filings, biomedical abstracts and user documents, "
        "with grounded answers, verified citations and evaluation endpoints."
    ),
    version=__version__,
    lifespan=lifespan,
)


def get_orchestrator() -> Orchestrator:
    if state.orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=state.error or "the index is not available",
        )
    return state.orchestrator


def get_index() -> RetrievalIndex:
    if state.index is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=state.error or "the index is not available",
        )
    return state.index


@app.get("/health", response_model=HealthResponse, tags=["operations"])
def health() -> HealthResponse:
    """Readiness, index statistics and the configured database."""
    index_info: dict[str, Any] = state.index.describe() if state.index else {"built": False}
    if state.error:
        index_info["error"] = state.error
    return HealthResponse(
        status="ok" if state.index is not None else "degraded",
        version=__version__,
        index=index_info,
        database=redact_url(state.settings.database_url) if state.settings else "",
    )


@app.get("/metrics", tags=["operations"])
def metrics() -> Response:
    """Prometheus exposition endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/query", response_model=AnswerResponse, tags=["retrieval"])
def query(
    request: QueryRequest, orchestrator: Orchestrator = Depends(get_orchestrator)
) -> AnswerResponse:
    """Answer a question with citations resolving to retrieved passages."""
    try:
        answer = orchestrator.answer(request.query, top_k=request.top_k)
    except IndexNotBuiltError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except RagPlatformError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    QUERY_COUNTER.labels(strategy=str(answer.diagnostics.get("strategy", "single_hop"))).inc()
    QUERY_LATENCY.observe(answer.latency_ms / 1000.0)
    GROUNDEDNESS.observe(answer.groundedness)
    if state.store is not None:
        state.store.log_query(answer)
    return AnswerResponse.from_answer(answer, include_evidence=request.include_evidence)


@app.post("/documents", response_model=IngestResponse, tags=["ingestion"])
def ingest(request: IngestRequest, index: RetrievalIndex = Depends(get_index)) -> IngestResponse:
    """Add documents to the index.

    The index is rebuilt rather than appended to, because the default embedding
    backend is fit on the corpus. The response reports the rebuild time so a
    caller batching uploads can see the cost.
    """
    documents = [
        Document.create(
            source_type=item.source_type,
            text=item.text,
            metadata=DocumentMetadata(title=item.title, **item.metadata),
        )
        for item in request.documents
    ]
    try:
        with Stopwatch() as sw:
            added = index.add_documents(documents)
    except RagPlatformError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if state.store is not None:
        state.store.upsert_documents(index.documents)
        state.store.upsert_chunks(index.chunks)
    if state.config is not None:
        state.orchestrator = Orchestrator(index, state.config)

    INGESTED.inc(added)
    return IngestResponse(
        ingested=added,
        total_documents=len(index.documents),
        total_chunks=len(index.chunks),
        rebuild_ms=round(sw.elapsed_ms, 2),
    )


@app.post("/evaluate", response_model=EvaluationResponse, tags=["evaluation"])
def evaluate(
    request: EvaluationRequest, index: RetrievalIndex = Depends(get_index)
) -> EvaluationResponse:
    """Score the live index against a judgement file.

    Exposing evaluation as an endpoint is what makes a quality regression
    detectable after deployment rather than only at benchmark time.
    """
    qrels_path = request.qrels_path or (
        str(state.config.corpus.qrels_path)
        if state.config and state.config.corpus.qrels_path
        else None
    )
    if not qrels_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="no judgement file configured"
        )
    try:
        qrels = load_qrels(qrels_path)
        result = evaluate_retrieval(index, qrels, top_k=request.top_k)
    except EvaluationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return EvaluationResponse(
        queries=result["queries"],
        metrics=result["metrics"],
        confidence_intervals=result.get("confidence_intervals", {}),
        by_query_type=result.get("by_query_type", {}),
        latency=result.get("latency", {}),
    )
