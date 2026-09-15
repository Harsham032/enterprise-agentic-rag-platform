"""Acquisition clients, exercised against recorded upstream payloads.

No test in this module touches the network. Real responses are recorded in
``tests/fixtures`` and replayed through an httpx mock transport, so the request
construction, the retry policy and the parsers are all covered offline.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest

from rag_platform.data.http import HttpSource, RateLimiter
from rag_platform.data.pubmed import PubMedClient, parse_pubmed_xml, summarise_records
from rag_platform.data.sec_edgar import (
    SecEdgarClient,
    extract_us_gaap_facts,
    normalise_cik,
    split_10k_sections,
)
from rag_platform.errors import DataAcquisitionError

# --------------------------------------------------------------- rate limiter


def test_rate_limiter_enforces_minimum_interval() -> None:
    limiter = RateLimiter(requests_per_second=50.0)
    start = time.monotonic()
    for _ in range(3):
        limiter.acquire()
    # Three acquisitions means two enforced gaps of 20ms.
    assert time.monotonic() - start >= 0.035


def test_rate_limiter_rejects_non_positive_rate() -> None:
    with pytest.raises(ValueError):
        RateLimiter(0)


# ------------------------------------------------------------------ retries


def test_retryable_status_is_retried_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="try later")
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    source = HttpSource(
        base_url="https://example.test", headers={}, requests_per_second=1000.0, client=client
    )
    assert source.get_json("/thing") == {"ok": True}
    assert calls["n"] == 3


def test_client_error_is_not_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404, text="missing")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    source = HttpSource(
        base_url="https://example.test", headers={}, requests_per_second=1000.0, client=client
    )
    with pytest.raises(DataAcquisitionError, match="404"):
        source.get_json("/thing")
    assert calls["n"] == 1


def test_transport_failure_becomes_a_domain_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("blocked", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    source = HttpSource(
        base_url="https://example.test", headers={}, requests_per_second=1000.0, client=client
    )
    with pytest.raises(DataAcquisitionError, match="could not reach"):
        source.get_json("/thing")


# ---------------------------------------------------------------- SEC EDGAR


def test_sec_client_requires_contact_user_agent() -> None:
    with pytest.raises(DataAcquisitionError, match="contact email"):
        SecEdgarClient("anonymous-crawler")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1750", "0000001750"), (1750, "0000001750"), ("CIK0000001750", "0000001750")],
)
def test_normalise_cik(raw: str | int, expected: str) -> None:
    assert normalise_cik(raw) == expected


def test_normalise_cik_rejects_non_numeric() -> None:
    with pytest.raises(DataAcquisitionError):
        normalise_cik("not-a-cik")


def _sec_client(fixtures_dir: Path) -> SecEdgarClient:
    submissions = json.loads((fixtures_dir / "sec_submissions.json").read_text())
    facts = json.loads((fixtures_dir / "sec_companyfacts.json").read_text())

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "submissions" in path:
            return httpx.Response(200, json=submissions)
        if "companyfacts" in path:
            return httpx.Response(200, json=facts)
        if "company_tickers" in path:
            return httpx.Response(
                200, json={"0": {"cik_str": 1750, "ticker": "EXI", "title": "Example"}}
            )
        return httpx.Response(
            200, text="<html><body><p>Item 1A. Risk Factors</p><p>Risks exist.</p></body></html>"
        )

    return SecEdgarClient(
        "Test Suite tests@example.com",
        requests_per_second=1000.0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_ticker_resolves_to_cik(fixtures_dir: Path) -> None:
    with _sec_client(fixtures_dir) as client:
        assert client.ticker_to_cik("exi") == "0000001750"


def test_unknown_ticker_raises(fixtures_dir: Path) -> None:
    with (
        _sec_client(fixtures_dir) as client,
        pytest.raises(DataAcquisitionError, match="ticker not present"),
    ):
        client.ticker_to_cik("ZZZZ")


def test_list_filings_transposes_and_filters(fixtures_dir: Path) -> None:
    with _sec_client(fixtures_dir) as client:
        filings = client.list_filings("1750", form_types=("10-K",), limit=5)
    assert [f["form"] for f in filings] == ["10-K", "10-K"]
    assert filings[0]["accession_plain"] == "000000175024000012"
    assert filings[0]["company_name"] == "EXAMPLE INDUSTRIES INC"


def test_filing_document_url_drops_cik_padding(fixtures_dir: Path) -> None:
    with _sec_client(fixtures_dir) as client:
        filing = client.list_filings("1750", limit=1)[0]
        url = client.filing_document_url(filing)
    assert url.endswith("/Archives/edgar/data/1750/000000175024000012/exi-20240531.htm")


def test_build_document_uses_report_date_for_fiscal_year(fixtures_dir: Path) -> None:
    with _sec_client(fixtures_dir) as client:
        filing = client.list_filings("1750", limit=1)[0]
    document = SecEdgarClient.build_document(
        filing, "Item 1A. Risk Factors\nSupply is concentrated."
    )
    assert document.metadata.fiscal_year == 2024
    assert document.metadata.ticker == "EXI"
    assert document.metadata.cik == "0000001750"


def test_company_facts_keeps_annual_and_latest_restatement(fixtures_dir: Path) -> None:
    with _sec_client(fixtures_dir) as client:
        rows = extract_us_gaap_facts(client.company_facts("1750"))
    revenues = {row["fiscal_year"]: row["value"] for row in rows if row["concept"] == "Revenues"}
    # The quarterly row is excluded and the later of the two 2024 filings wins.
    assert revenues == {2023: 5200000000.0, 2024: 5600000000.0}
    assert any(row["concept"] == "NetIncomeLoss" for row in rows)


def test_split_10k_sections_finds_items() -> None:
    text = "Item 1. Business\nWe build things.\n\nItem 1A. Risk Factors\nThings break.\n"
    sections = split_10k_sections(text)
    assert [section.heading for section in sections] == [
        "Item 1. Business",
        "Item 1A. Risk Factors",
    ]


def test_split_10k_sections_falls_back_to_full_text() -> None:
    sections = split_10k_sections("A filing exhibit with no item headings at all.")
    assert len(sections) == 1
    assert sections[0].heading == "Full Text"


# -------------------------------------------------------------------- PubMed


def test_pubmed_requires_contact_email() -> None:
    with pytest.raises(DataAcquisitionError, match="contact email"):
        PubMedClient(tool="tests", email="")


def test_parse_pubmed_xml_keeps_structured_sections(fixtures_dir: Path) -> None:
    documents = parse_pubmed_xml((fixtures_dir / "pubmed_efetch.xml").read_text())
    # The record without an abstract is dropped: it cannot ground an answer.
    assert len(documents) == 2
    structured = documents[0]
    assert structured.metadata.pmid == "35000001"
    assert [section.heading for section in structured.sections] == [
        "Background",
        "Methods",
        "Results",
    ]
    assert structured.metadata.authors == ["Rivera A", "Chen W"]
    assert structured.metadata.publication_year == 2022
    assert structured.metadata.mesh_terms == ["Randomized Controlled Trials as Topic"]


def test_parse_pubmed_xml_handles_medline_date(fixtures_dir: Path) -> None:
    documents = parse_pubmed_xml((fixtures_dir / "pubmed_efetch.xml").read_text())
    unstructured = documents[1]
    assert unstructured.metadata.publication_year == 2021
    assert [section.heading for section in unstructured.sections] == ["Abstract"]


def test_parse_pubmed_xml_rejects_malformed_payload() -> None:
    with pytest.raises(DataAcquisitionError, match="malformed"):
        parse_pubmed_xml("<PubmedArticleSet><broken>")


def test_pubmed_search_and_fetch(fixtures_dir: Path) -> None:
    xml = (fixtures_dir / "pubmed_efetch.xml").read_text()

    def handler(request: httpx.Request) -> httpx.Response:
        if "esearch" in request.url.path:
            assert request.url.params["tool"] == "tests"
            assert request.url.params["email"] == "tests@example.com"
            return httpx.Response(200, json={"esearchresult": {"idlist": ["35000001", "35000002"]}})
        assert request.url.params["id"] == "35000001,35000002"
        return httpx.Response(200, text=xml)

    client = PubMedClient(
        tool="tests",
        email="tests@example.com",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    documents = client.search_and_fetch("anything", retmax=2)
    assert len(documents) == 2
    assert summarise_records(documents)["records"] == 2


def test_pubmed_rejects_unexpected_search_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    client = PubMedClient(
        tool="tests",
        email="tests@example.com",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(DataAcquisitionError, match="unexpected esearch"):
        client.search("anything")
