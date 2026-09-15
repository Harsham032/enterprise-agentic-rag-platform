"""Shared fixtures.

The repository's bundled evaluation corpus doubles as the test corpus: keeping
one corpus means a test failure and a benchmark regression point at the same
data, and the tests exercise the same ingestion path the benchmark does.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rag_platform.config import PipelineConfig  # noqa: E402
from rag_platform.data.local_documents import load_directory  # noqa: E402
from rag_platform.data.models import Document, DocumentMetadata, Section, SourceType  # noqa: E402
from rag_platform.retrieval.index import RetrievalIndex  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def config(repo_root: Path) -> PipelineConfig:
    return PipelineConfig.from_yaml(repo_root / "configs" / "default.yaml")


@pytest.fixture(scope="session")
def corpus(repo_root: Path) -> list[Document]:
    return load_directory(repo_root / "data" / "fixtures" / "corpus")


@pytest.fixture(scope="session")
def index(config: PipelineConfig, corpus: list[Document]) -> RetrievalIndex:
    """A built index shared across tests; treat it as read-only."""
    built = RetrievalIndex(config)
    built.build_from_documents(corpus)
    return built


@pytest.fixture
def sample_document() -> Document:
    body = (
        "We depend on a concentrated set of turbine suppliers. Two manufacturers "
        "supplied 84 percent of the turbines in our operating fleet. A prolonged "
        "production interruption would delay our construction programme and "
        "increase the cost of replacement components significantly."
    )
    second = (
        "Total revenue for 2024 was 1,284.6 million dollars, an increase of 4.2 "
        "percent. Adjusted EBITDA was 742.1 million dollars for the same period."
    )
    return Document.create(
        source_type=SourceType.SEC_FILING,
        text=f"{body}\n\n{second}",
        metadata=DocumentMetadata(
            title="Example Filing",
            company_name="Example Energy",
            form_type="10-K",
            fiscal_year=2024,
        ),
        sections=[
            Section(heading="Item 1A. Risk Factors", text=body, order=0),
            Section(heading="Item 7. Management's Discussion and Analysis", text=second, order=1),
        ],
        natural_key="example-filing-2024",
    )
