"""HTTP surface: health, query, ingestion, evaluation and metrics."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from rag_platform.services.api import app


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_a_built_index(client: TestClient) -> None:
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["index"]["built"] is True
    assert payload["index"]["documents"] > 0
    assert payload["index"]["chunks"] > payload["index"]["documents"]


def test_health_redacts_database_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """The health endpoint names the configured database; it must not publish its password."""
    from rag_platform.data.store import redact_url

    redacted = redact_url("postgresql+psycopg://rag:hunter2@db.internal:5432/rag_platform")
    assert "hunter2" not in redacted
    assert redacted == "postgresql+psycopg://rag:***@db.internal:5432/rag_platform"


def test_health_body_carries_no_credential_values(client: TestClient) -> None:
    body = client.get("/health").text
    assert "api_key" not in body
    assert "password" not in body
    assert "@" not in body


def test_query_returns_a_cited_answer(client: TestClient) -> None:
    response = client.post(
        "/query",
        json={
            "query": "Which turbine supplier concentration risk does Northwind Energy disclose?",
            "top_k": 5,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert payload["citations"]
    assert payload["latency_ms"] > 0
    # Every citation must point at a chunk that was actually returned.
    evidence_ids = {chunk["chunk_id"] for chunk in payload["evidence"]}
    assert all(citation["chunk_id"] in evidence_ids for citation in payload["citations"])


def test_query_can_suppress_the_evidence_payload(client: TestClient) -> None:
    payload = client.post(
        "/query", json={"query": "revenue growth", "include_evidence": False}
    ).json()
    assert payload["evidence"] == []
    assert payload["answer"]


def test_query_reports_decomposition_diagnostics(client: TestClient) -> None:
    payload = client.post(
        "/query", json={"query": "Compare Northwind Energy revenue growth in 2024 against 2023"}
    ).json()
    assert payload["diagnostics"]["strategy"] == "temporal_comparison"
    assert len(payload["subqueries"]) == 2
    assert {"planning_ms", "retrieval_ms", "generation_ms"} <= set(payload["diagnostics"])


def test_empty_query_is_rejected(client: TestClient) -> None:
    assert client.post("/query", json={"query": ""}).status_code == 422


def test_out_of_range_top_k_is_rejected(client: TestClient) -> None:
    assert client.post("/query", json={"query": "revenue", "top_k": 500}).status_code == 422


def test_evaluate_scores_the_live_index(client: TestClient) -> None:
    payload = client.post("/evaluate", json={}).json()
    assert payload["queries"] > 0
    assert 0.0 <= payload["metrics"]["recall@budget"] <= 1.0
    assert "recall@budget" in payload["confidence_intervals"]
    assert payload["by_query_type"]


def test_evaluate_rejects_a_missing_judgement_file(client: TestClient) -> None:
    response = client.post("/evaluate", json={"qrels_path": "data/fixtures/nope.json"})
    assert response.status_code == 400


def test_ingestion_adds_documents_and_makes_them_retrievable(client: TestClient) -> None:
    before = client.get("/health").json()["index"]["documents"]
    response = client.post(
        "/documents",
        json={
            "documents": [
                {
                    "title": "Scheduled Maintenance Window",
                    "text": (
                        "The quarterly maintenance window runs on the first Sunday of each "
                        "month between 02:00 and 05:00 UTC. Ingestion is paused for the "
                        "duration and queued documents are processed afterwards."
                    ),
                }
            ]
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ingested"] == 1
    assert payload["total_documents"] == before + 1

    answer = client.post(
        "/query", json={"query": "When does the quarterly maintenance window run?"}
    ).json()
    assert "first Sunday" in answer["answer"]


def test_ingestion_rejects_an_empty_batch(client: TestClient) -> None:
    assert client.post("/documents", json={"documents": []}).status_code == 422


def test_metrics_endpoint_exposes_counters(client: TestClient) -> None:
    client.post("/query", json={"query": "revenue growth"})
    body = client.get("/metrics").text
    assert "rag_queries_total" in body
    assert "rag_query_latency_seconds" in body
    assert "rag_answer_groundedness" in body


def test_openapi_document_is_served(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert set(schema["paths"]) >= {"/health", "/query", "/documents", "/evaluate", "/metrics"}
