"""Configuration model.

Two layers are merged, in increasing order of precedence:

1. a YAML file describing the pipeline (chunking, retrieval, evaluation);
2. ``RAG_*`` environment variables describing the deployment (databases,
   credentials, service ports), loaded from ``.env`` when present.

Splitting them this way keeps experiment configuration reviewable in version
control while secrets stay out of the repository.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .errors import ConfigurationError

EmbeddingBackend = Literal["tfidf_svd", "sentence_transformers", "openai"]
GenerationBackend = Literal["extractive", "anthropic", "openai"]
VectorBackend = Literal["numpy", "pgvector", "qdrant"]


class RunConfig(BaseModel):
    name: str = "default"
    seed: int = 20260101
    output_dir: Path = Path("reports")


class CorpusConfig(BaseModel):
    source_dir: Path = Path("data/fixtures/corpus")
    qrels_path: Path | None = Path("data/fixtures/qrels.json")


class ChunkingConfig(BaseModel):
    strategy: Literal["fixed_window", "section_aware"] = "section_aware"
    target_tokens: int = Field(default=180, gt=0)
    overlap_tokens: int = Field(default=40, ge=0)
    min_tokens: int = Field(default=40, ge=1)

    @model_validator(mode="after")
    def _overlap_below_target(self) -> ChunkingConfig:
        if self.overlap_tokens >= self.target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens")
        return self


class EmbeddingConfig(BaseModel):
    backend: EmbeddingBackend = "tfidf_svd"
    dim: int = Field(default=256, gt=0)
    min_df: int = 1
    max_df: float = Field(default=0.85, gt=0.0, le=1.0)
    ngram_max: int = Field(default=2, ge=1, le=3)


class RetrievalConfig(BaseModel):
    top_k: int = Field(default=10, gt=0)
    candidate_k: int = Field(default=50, gt=0)
    fusion: Literal["weighted", "rrf"] = "weighted"
    lexical_weight: float = Field(default=0.45, ge=0.0, le=1.0)
    dense_weight: float = Field(default=0.55, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, gt=0)
    stemming: bool = True

    @model_validator(mode="after")
    def _weights_are_usable(self) -> RetrievalConfig:
        if self.fusion == "weighted" and self.lexical_weight + self.dense_weight == 0:
            raise ValueError("weighted fusion needs at least one non-zero weight")
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be >= top_k")
        return self


class RerankingConfig(BaseModel):
    enabled: bool = True
    model: Literal["logistic", "none"] = "logistic"
    candidate_k: int = Field(default=30, gt=0)
    top_k: int = Field(default=10, gt=0)


class PlanningConfig(BaseModel):
    enabled: bool = True
    max_subqueries: int = Field(default=3, ge=1, le=8)


class GenerationConfig(BaseModel):
    backend: GenerationBackend = "extractive"
    # Number of evidence spans quoted in an extractive answer.
    max_spans: int = Field(default=4, gt=0)
    # Sentences per span. Lead-in sentences ("Three alerts page the on-call
    # engineer.") carry the query vocabulary while the detail sits in the
    # sentences that follow, so a span of one systematically truncates answers.
    span_sentences: int = Field(default=2, ge=1, le=5)
    min_support_overlap: float = Field(default=0.35, ge=0.0, le=1.0)


class EvaluationConfig(BaseModel):
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    primary_k: int = 5
    # Evidence budget in whitespace tokens. Metrics computed at a fixed token
    # budget are the only ones comparable across chunk sizes, because "top 5
    # chunks" returns three times as much text at 250-token chunks as at
    # 75-token chunks.
    token_budget: int = Field(default=400, gt=0)
    max_budget_k: int = Field(default=25, gt=0)
    latency_warmup: int = Field(default=2, ge=0)

    @field_validator("k_values")
    @classmethod
    def _sorted_positive(cls, value: list[int]) -> list[int]:
        if not value or any(k <= 0 for k in value):
            raise ValueError("k_values must be a non-empty list of positive integers")
        return sorted(set(value))

    @model_validator(mode="after")
    def _primary_k_present(self) -> EvaluationConfig:
        if self.primary_k not in self.k_values:
            raise ValueError("primary_k must be one of k_values")
        return self


class PipelineConfig(BaseModel):
    """Experiment configuration loaded from YAML."""

    run: RunConfig = Field(default_factory=RunConfig)
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    planning: PlanningConfig = Field(default_factory=PlanningConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load a configuration file, raising :class:`ConfigurationError` on bad input."""
        config_path = Path(path)
        if not config_path.is_file():
            raise ConfigurationError(f"configuration file not found: {config_path}")
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:  # pragma: no cover - depends on malformed input
            raise ConfigurationError(f"could not parse {config_path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigurationError(f"{config_path} must contain a YAML mapping")
        try:
            return cls.model_validate(raw)
        except Exception as exc:
            raise ConfigurationError(f"invalid configuration in {config_path}: {exc}") from exc

    def with_overrides(self, overrides: dict[str, Any]) -> PipelineConfig:
        """Return a copy with ``dotted.key=value`` overrides applied.

        Values are parsed as YAML scalars so ``retrieval.top_k=20`` yields an
        integer and ``reranking.enabled=false`` yields a boolean.
        """
        if not overrides:
            return self
        merged = self.model_dump(mode="python")
        for dotted_key, value in overrides.items():
            parts = dotted_key.split(".")
            cursor: Any = merged
            for part in parts[:-1]:
                if not isinstance(cursor, dict) or part not in cursor:
                    raise ConfigurationError(f"unknown configuration key: {dotted_key}")
                cursor = cursor[part]
            leaf = parts[-1]
            if not isinstance(cursor, dict) or leaf not in cursor:
                raise ConfigurationError(f"unknown configuration key: {dotted_key}")
            cursor[leaf] = yaml.safe_load(value) if isinstance(value, str) else value
        try:
            return PipelineConfig.model_validate(merged)
        except Exception as exc:
            raise ConfigurationError(f"invalid override: {exc}") from exc


class Settings(BaseSettings):
    """Deployment settings sourced from the environment."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    env: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    database_url: str = "sqlite:///data/processed/rag_platform.sqlite3"
    vector_backend: VectorBackend = "numpy"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "rag_chunks"

    redis_url: str = "redis://localhost:6379/0"
    cache_enabled: bool = False

    embedding_backend: EmbeddingBackend = "tfidf_svd"
    embedding_dim: int = 256
    sentence_transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    generation_backend: GenerationBackend = "extractive"

    sec_user_agent: str = ""
    ncbi_tool_name: str = "enterprise-rag-platform"
    ncbi_email: str = ""

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    ncbi_api_key: str = ""

    @property
    def is_production(self) -> bool:
        return self.env.lower() in {"prod", "production"}


def load_settings() -> Settings:
    """Read deployment settings from the environment."""
    return Settings()
