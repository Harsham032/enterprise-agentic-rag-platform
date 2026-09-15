"""SEC EDGAR acquisition.

Endpoints used (all public, no key required):

* ``https://www.sec.gov/files/company_tickers.json`` - ticker to CIK map.
* ``https://data.sec.gov/submissions/CIK##########.json`` - filing history.
* ``https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`` - XBRL facts.
* ``https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}`` - the
  filing documents themselves.

The SEC requires a descriptive ``User-Agent`` carrying a contact address and
asks automated clients to stay at or below ten requests per second. Both are
enforced by this client; see https://www.sec.gov/os/webmaster-faq#developers.
"""

from __future__ import annotations

import re
from typing import Any

import httpx

from ..errors import DataAcquisitionError
from ..logging_utils import get_logger
from ..utils.text import normalise_whitespace
from .http import HttpSource
from .models import Document, DocumentMetadata, Section, SourceType

logger = get_logger(__name__)

SEC_ARCHIVES_BASE = "https://www.sec.gov"
SEC_DATA_BASE = "https://data.sec.gov"
SEC_RATE_LIMIT_PER_SECOND = 8.0  # below the published ceiling of 10/s

# The canonical Item headings of a Form 10-K. Segmenting on them turns a single
# multi-hundred-page filing into sections that chunking and citations can point
# at precisely.
TENK_ITEM_PATTERN = re.compile(
    r"^\s*item\s+(\d{1,2}[ab]?)\s*[.:\-–]?\s*(.{0,120}?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

KNOWN_ITEM_TITLES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "2": "Properties",
    "3": "Legal Proceedings",
    "5": "Market for Registrant's Common Equity",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9A": "Controls and Procedures",
}


def normalise_cik(cik: str | int) -> str:
    """Return the ten-digit zero-padded CIK used by the EDGAR JSON APIs."""
    digits = re.sub(r"\D", "", str(cik))
    if not digits:
        raise DataAcquisitionError(f"not a usable CIK: {cik!r}")
    return digits.zfill(10)


def split_10k_sections(text: str) -> list[Section]:
    """Segment 10-K body text on its Item headings.

    Returns a single ``Full Text`` section when no headings are found, which is
    the common case for exhibits and for filings whose table of contents uses
    non-standard wording.
    """
    matches = list(TENK_ITEM_PATTERN.finditer(text))
    if not matches:
        cleaned = normalise_whitespace(text)
        return [Section(heading="Full Text", text=cleaned, order=0)] if cleaned else []

    sections: list[Section] = []
    for order, match in enumerate(matches):
        item_number = match.group(1).upper()
        inline_title = match.group(2).strip(" .:-")
        title = inline_title or KNOWN_ITEM_TITLES.get(item_number, "")
        heading = f"Item {item_number}. {title}".strip().rstrip(".")
        start = match.end()
        end = matches[order + 1].start() if order + 1 < len(matches) else len(text)
        body = normalise_whitespace(text[start:end])
        if body:
            sections.append(Section(heading=heading, text=body, order=order))
    return sections


class SecEdgarClient(HttpSource):
    """Client for the public EDGAR endpoints."""

    def __init__(
        self,
        user_agent: str,
        *,
        requests_per_second: float = SEC_RATE_LIMIT_PER_SECOND,
        client: httpx.Client | None = None,
    ) -> None:
        if not user_agent or "@" not in user_agent:
            raise DataAcquisitionError(
                "SEC EDGAR requires a User-Agent containing a contact email address; "
                "set RAG_SEC_USER_AGENT, for example 'Jane Doe jane@example.com'"
            )
        super().__init__(
            base_url=SEC_DATA_BASE,
            headers={
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Host": "data.sec.gov",
            },
            requests_per_second=requests_per_second,
            client=client,
        )
        self.user_agent = user_agent

    def ticker_to_cik(self, ticker: str) -> str:
        """Resolve an exchange ticker to its zero-padded CIK."""
        payload = self.get_json(f"{SEC_ARCHIVES_BASE}/files/company_tickers.json")
        wanted = ticker.strip().upper()
        for entry in payload.values():
            if str(entry.get("ticker", "")).upper() == wanted:
                return normalise_cik(entry["cik_str"])
        raise DataAcquisitionError(f"ticker not present in the EDGAR ticker map: {ticker}")

    def company_submissions(self, cik: str | int) -> dict[str, Any]:
        """Return the submissions record, which carries the recent filing index."""
        return self.get_json(f"/submissions/CIK{normalise_cik(cik)}.json")

    def company_facts(self, cik: str | int) -> dict[str, Any]:
        """Return the full XBRL company-facts record used by structured lookups."""
        return self.get_json(f"/api/xbrl/companyfacts/CIK{normalise_cik(cik)}.json")

    def list_filings(
        self,
        cik: str | int,
        *,
        form_types: tuple[str, ...] = ("10-K",),
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Return filing index entries for ``cik`` restricted to ``form_types``.

        The ``recent`` block of the submissions record is column-oriented; it is
        transposed here into one record per filing.
        """
        submissions = self.company_submissions(cik)
        recent = submissions.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        wanted = {form.upper() for form in form_types}

        results: list[dict[str, Any]] = []
        for index, form in enumerate(forms):
            if form.upper() not in wanted:
                continue
            accession = recent["accessionNumber"][index]
            results.append(
                {
                    "accession_number": accession,
                    "accession_plain": accession.replace("-", ""),
                    "form": form,
                    "filing_date": recent["filingDate"][index],
                    "report_date": recent.get("reportDate", [None] * len(forms))[index],
                    "primary_document": recent["primaryDocument"][index],
                    "cik": normalise_cik(cik),
                    "company_name": submissions.get("name", ""),
                    "tickers": submissions.get("tickers", []),
                }
            )
            if len(results) >= limit:
                break
        return results

    def filing_document_url(self, filing: dict[str, Any]) -> str:
        """Build the archive URL of a filing's primary document."""
        cik_int = int(filing["cik"])
        return (
            f"{SEC_ARCHIVES_BASE}/Archives/edgar/data/{cik_int}/"
            f"{filing['accession_plain']}/{filing['primary_document']}"
        )

    def fetch_filing_html(self, filing: dict[str, Any]) -> str:
        """Download the raw primary document for a filing index entry."""
        return self.get_text(self.filing_document_url(filing))

    @staticmethod
    def build_document(filing: dict[str, Any], body_text: str) -> Document:
        """Assemble a :class:`Document` from a filing entry and its extracted text."""
        filing_date = filing.get("filing_date")
        report_date = filing.get("report_date") or filing_date
        fiscal_year = int(str(report_date)[:4]) if report_date else None
        tickers = filing.get("tickers") or []
        metadata = DocumentMetadata(
            title=f"{filing.get('company_name', '')} {filing.get('form', '')} {fiscal_year or ''}".strip(),
            company_name=filing.get("company_name"),
            ticker=tickers[0] if tickers else None,
            cik=filing.get("cik"),
            form_type=filing.get("form"),
            fiscal_year=fiscal_year,
            filing_date=filing_date,
            url=filing.get("document_url"),
        )
        sections = split_10k_sections(body_text)
        return Document.create(
            source_type=SourceType.SEC_FILING,
            text=normalise_whitespace(body_text),
            metadata=metadata,
            sections=sections,
            natural_key=filing["accession_number"],
        )


def extract_us_gaap_facts(
    company_facts: dict[str, Any],
    concepts: tuple[str, ...] = ("Revenues", "NetIncomeLoss", "Assets", "Liabilities"),
) -> list[dict[str, Any]]:
    """Flatten selected us-gaap concepts into rows for the structured store.

    Only annual figures (``form`` starting with ``10-K`` and a ``fy`` present)
    are kept; quarterly and amended restatements would otherwise produce several
    conflicting values for the same fiscal year.
    """
    facts = company_facts.get("facts", {}).get("us-gaap", {})
    entity_name = company_facts.get("entityName", "")
    cik = normalise_cik(company_facts.get("cik", 0)) if company_facts.get("cik") else None

    rows: list[dict[str, Any]] = []
    for concept in concepts:
        concept_block = facts.get(concept)
        if not concept_block:
            continue
        for unit, entries in concept_block.get("units", {}).items():
            for entry in entries:
                if not str(entry.get("form", "")).startswith("10-K"):
                    continue
                if entry.get("fy") is None or entry.get("fp") != "FY":
                    continue
                rows.append(
                    {
                        "cik": cik,
                        "entity_name": entity_name,
                        "concept": concept,
                        "unit": unit,
                        "fiscal_year": int(entry["fy"]),
                        "value": float(entry["val"]),
                        "end_date": entry.get("end"),
                        "accession_number": entry.get("accn"),
                    }
                )
    # Later filings supersede earlier ones for the same (concept, year).
    deduped: dict[tuple[str, int], dict[str, Any]] = {}
    for row in sorted(rows, key=lambda r: (r["concept"], r["fiscal_year"], r["end_date"] or "")):
        deduped[(row["concept"], row["fiscal_year"])] = row
    return list(deduped.values())
