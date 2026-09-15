"""Relational persistence for documents, chunks, facts and the query log."""

from __future__ import annotations

import pytest

from rag_platform.data.models import Answer, Document, DocumentMetadata, SourceType
from rag_platform.data.store import MetadataStore, postgres_schema_sql
from rag_platform.features.chunking import chunk_documents
from rag_platform.features.metadata import enrich_metadata


@pytest.fixture
def store() -> MetadataStore:
    created = MetadataStore("sqlite:///:memory:")
    created.create_all()
    yield created
    created.close()


def test_documents_and_chunks_round_trip(store: MetadataStore, corpus, config) -> None:
    documents = [enrich_metadata(document) for document in corpus]
    assert store.upsert_documents(documents) == len(documents)
    chunks = chunk_documents(documents, config.chunking)
    assert store.upsert_chunks(chunks) == len(chunks)
    assert store.document_count() == len(documents)
    assert store.chunk_count() == len(chunks)


def test_reingesting_replaces_rather_than_duplicates(store: MetadataStore, corpus, config) -> None:
    documents = [enrich_metadata(document) for document in corpus]
    chunks = chunk_documents(documents, config.chunking)
    for _ in range(2):
        store.upsert_documents(documents)
        store.upsert_chunks(chunks)
    assert store.document_count() == len(documents)
    assert store.chunk_count() == len(chunks)


def test_upserting_nothing_is_a_no_op(store: MetadataStore) -> None:
    assert store.upsert_documents([]) == 0
    assert store.upsert_chunks([]) == 0
    assert store.upsert_financial_facts([]) == 0


def test_financial_facts_lookup_filters(store: MetadataStore) -> None:
    rows = [
        {
            "cik": "0000001750",
            "concept": "Revenues",
            "fiscal_year": 2024,
            "unit": "USD",
            "value": 5.6e9,
        },
        {
            "cik": "0000001750",
            "concept": "Revenues",
            "fiscal_year": 2023,
            "unit": "USD",
            "value": 5.2e9,
        },
        {
            "cik": "0000009999",
            "concept": "Revenues",
            "fiscal_year": 2024,
            "unit": "USD",
            "value": 1.0e9,
        },
    ]
    assert store.upsert_financial_facts(rows) == 3
    assert len(store.lookup_facts("0000001750")) == 2
    assert len(store.lookup_facts("0000001750", 2024)) == 1
    assert store.lookup_facts("0000001750", 2024)[0]["value"] == pytest.approx(5.6e9)
    assert store.lookup_facts(concepts=["NetIncomeLoss"]) == []


def test_financial_facts_upsert_replaces_by_key(store: MetadataStore) -> None:
    row = {
        "cik": "0000001750",
        "concept": "Revenues",
        "fiscal_year": 2024,
        "unit": "USD",
        "value": 1.0,
    }
    store.upsert_financial_facts([row])
    store.upsert_financial_facts([{**row, "value": 2.0}])
    assert store.lookup_facts("0000001750", 2024)[0]["value"] == pytest.approx(2.0)


def test_query_log_records_a_served_answer(store: MetadataStore) -> None:
    answer = Answer(
        query="what was revenue",
        text="Revenue rose. [1]",
        groundedness=0.9,
        latency_ms=12.3,
        tools_used=["document_search"],
        diagnostics={"strategy": "single_hop"},
    )
    query_id = store.log_query(answer)
    recent = store.recent_queries()
    assert len(recent) == 1
    assert recent[0]["query_id"] == query_id
    assert recent[0]["strategy"] == "single_hop"
    assert recent[0]["groundedness"] == pytest.approx(0.9)


def test_sqlite_file_directory_is_created(tmp_path) -> None:
    nested = tmp_path / "a" / "b" / "db.sqlite3"
    created = MetadataStore(f"sqlite:///{nested}")
    created.create_all()
    assert nested.exists()
    created.close()


def test_drop_all_removes_tables(store: MetadataStore) -> None:
    document = Document.create(
        source_type=SourceType.USER_DOCUMENT,
        text="text",
        metadata=DocumentMetadata(title="t"),
        natural_key="k",
    )
    store.upsert_documents([document])
    store.drop_all()
    store.create_all()
    assert store.document_count() == 0


def test_postgres_schema_is_shipped_and_covers_every_table() -> None:
    sql = postgres_schema_sql()
    for table in ("documents", "chunks", "financial_facts", "query_log"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in sql


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("sqlite:///data/db.sqlite3", "sqlite:///data/db.sqlite3"),
        (
            "postgresql+psycopg://rag:hunter2@db:5432/rag",
            "postgresql+psycopg://rag:***@db:5432/rag",
        ),
        ("postgresql://user@host/db", "postgresql://user@host/db"),
    ],
)
def test_redact_url_strips_passwords(url: str, expected: str) -> None:
    from rag_platform.data.store import redact_url

    assert redact_url(url) == expected
