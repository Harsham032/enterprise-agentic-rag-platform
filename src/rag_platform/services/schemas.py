"""Request and response models for the HTTP API.

Kept separate from the domain model in :mod:`rag_platform.data.models` so the
wire format can stay stable while internal structures change.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..data.models import Answer, ScoredChunk, SourceType


class QueryRequest(BaseModel):
    """A question for the platform to answer."""

    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    source_types: list[SourceType] | None = None
    include_evidence: bool = True


class CitationResponse(BaseModel):
    marker: int
    chunk_id: str
    doc_id: str
    label: str
    quote: str
    support_score: float
    verified: bool


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    doc_id: str
    source_type: str
    section_heading: str | None
    text: str
    score: float
    lexical_score: float
    dense_score: float
    rerank_score: float | None
    rank: int

    @classmethod
    def from_scored(cls, scored: ScoredChunk) -> RetrievedChunkResponse:
        return cls(
            chunk_id=scored.chunk_id,
            doc_id=scored.chunk.doc_id,
            source_type=scored.chunk.source_type.value,
            section_heading=scored.chunk.section_heading,
            text=scored.chunk.text,
            score=scored.score,
            lexical_score=scored.lexical_score,
            dense_score=scored.dense_score,
            rerank_score=scored.rerank_score,
            rank=scored.rank,
        )


class AnswerResponse(BaseModel):
    """A cited answer plus the diagnostics needed to debug a bad one."""

    query: str
    answer: str
    citations: list[CitationResponse]
    evidence: list[RetrievedChunkResponse] = Field(default_factory=list)
    subqueries: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    groundedness: float
    latency_ms: float
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_answer(cls, answer: Answer, *, include_evidence: bool = True) -> AnswerResponse:
        return cls(
            query=answer.query,
            answer=answer.text,
            citations=[CitationResponse(**citation.model_dump()) for citation in answer.citations],
            evidence=(
                [RetrievedChunkResponse.from_scored(scored) for scored in answer.supporting_chunks]
                if include_evidence
                else []
            ),
            subqueries=answer.subqueries,
            tools_used=answer.tools_used,
            groundedness=answer.groundedness,
            latency_ms=answer.latency_ms,
            diagnostics=answer.diagnostics,
        )


class IngestRequest(BaseModel):
    """Documents supplied directly rather than acquired from a source."""

    documents: list[InlineDocument] = Field(min_length=1, max_length=200)


class InlineDocument(BaseModel):
    text: str = Field(min_length=1)
    title: str = ""
    source_type: SourceType = SourceType.USER_DOCUMENT
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    ingested: int
    total_documents: int
    total_chunks: int
    rebuild_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    index: dict[str, Any]
    database: str


class EvaluationRequest(BaseModel):
    qrels_path: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)


class EvaluationResponse(BaseModel):
    queries: int
    metrics: dict[str, float]
    confidence_intervals: dict[str, dict[str, float]] = Field(default_factory=dict)
    by_query_type: dict[str, dict[str, float]] = Field(default_factory=dict)
    latency: dict[str, float] = Field(default_factory=dict)


IngestRequest.model_rebuild()
