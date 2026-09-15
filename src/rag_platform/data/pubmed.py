"""PubMed acquisition through the NCBI E-utilities.

Endpoints used:

* ``esearch.fcgi`` - resolve a query into a list of PMIDs.
* ``efetch.fcgi`` - retrieve the MEDLINE XML records for those PMIDs.

NCBI asks every automated client to identify itself with ``tool`` and ``email``
parameters and limits unauthenticated callers to three requests per second (ten
with an API key). Both are handled here; see
https://www.ncbi.nlm.nih.gov/books/NBK25497/.

Only titles, abstracts and bibliographic metadata are retrieved. Full text is
not fetched: most of it sits behind publisher licences that forbid
redistribution, and the abstract carries the retrievable signal for this task.
"""

from __future__ import annotations

from typing import Any

import httpx
from lxml import etree

from ..errors import DataAcquisitionError
from ..logging_utils import get_logger
from ..utils.text import normalise_whitespace
from .http import HttpSource
from .models import Document, DocumentMetadata, Section, SourceType

logger = get_logger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
RATE_LIMIT_WITHOUT_KEY = 2.5
RATE_LIMIT_WITH_KEY = 8.0
EFETCH_BATCH_SIZE = 100


class PubMedClient(HttpSource):
    """Client for the NCBI E-utilities restricted to the PubMed database."""

    def __init__(
        self,
        *,
        tool: str,
        email: str,
        api_key: str = "",
        client: httpx.Client | None = None,
    ) -> None:
        if not email or "@" not in email:
            raise DataAcquisitionError(
                "NCBI E-utilities require a contact email address; set RAG_NCBI_EMAIL"
            )
        super().__init__(
            base_url=EUTILS_BASE,
            headers={"User-Agent": f"{tool} ({email})"},
            requests_per_second=RATE_LIMIT_WITH_KEY if api_key else RATE_LIMIT_WITHOUT_KEY,
            client=client,
        )
        self.tool = tool
        self.email = email
        self.api_key = api_key

    def _common_params(self) -> dict[str, str]:
        params = {"tool": self.tool, "email": self.email, "db": "pubmed"}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def search(
        self, term: str, *, retmax: int = 50, mindate: str = "", maxdate: str = ""
    ) -> list[str]:
        """Return PMIDs matching ``term`` in PubMed relevance order."""
        params = self._common_params() | {"term": term, "retmax": str(retmax), "retmode": "json"}
        if mindate and maxdate:
            params |= {"mindate": mindate, "maxdate": maxdate, "datetype": "pdat"}
        payload = self.get_json("/esearch.fcgi", params)
        try:
            return list(payload["esearchresult"]["idlist"])
        except (KeyError, TypeError) as exc:
            raise DataAcquisitionError(f"unexpected esearch payload for {term!r}") from exc

    def fetch_records(self, pmids: list[str]) -> list[Document]:
        """Fetch and parse MEDLINE XML for ``pmids`` in batches."""
        documents: list[Document] = []
        for start in range(0, len(pmids), EFETCH_BATCH_SIZE):
            batch = pmids[start : start + EFETCH_BATCH_SIZE]
            params = self._common_params() | {"id": ",".join(batch), "retmode": "xml"}
            xml = self.get_text("/efetch.fcgi", params)
            documents.extend(parse_pubmed_xml(xml))
        return documents

    def search_and_fetch(self, term: str, *, retmax: int = 50) -> list[Document]:
        """Convenience wrapper combining :meth:`search` and :meth:`fetch_records`."""
        pmids = self.search(term, retmax=retmax)
        logger.info("pubmed_search", term=term, hits=len(pmids))
        return self.fetch_records(pmids) if pmids else []


def _element_text(node: etree._Element | None) -> str:
    """Return the concatenated text of a node, including inline markup."""
    if node is None:
        return ""
    return normalise_whitespace("".join(node.itertext()))


def parse_pubmed_xml(xml: str) -> list[Document]:
    """Parse a MEDLINE ``PubmedArticleSet`` payload into documents.

    Structured abstracts (``<AbstractText Label="METHODS">``) keep their labels
    as sections so retrieval can target, for example, the results of a trial
    rather than its background.
    """
    try:
        root = etree.fromstring(xml.encode("utf-8") if isinstance(xml, str) else xml)
    except etree.XMLSyntaxError as exc:
        raise DataAcquisitionError(f"malformed PubMed XML: {exc}") from exc

    documents: list[Document] = []
    for article in root.iter("PubmedArticle"):
        pmid = _element_text(article.find(".//PMID"))
        title = _element_text(article.find(".//Article/ArticleTitle"))

        sections: list[Section] = []
        abstract_parts: list[str] = []
        for order, node in enumerate(article.findall(".//Article/Abstract/AbstractText")):
            body = _element_text(node)
            if not body:
                continue
            label = (node.get("Label") or node.get("NlmCategory") or "").strip().title()
            heading = label or "Abstract"
            sections.append(Section(heading=heading, text=body, order=order))
            abstract_parts.append(f"{heading}: {body}" if label else body)

        if not abstract_parts:
            # Records without an abstract cannot support grounded answers.
            continue

        authors = []
        for author in article.findall(".//Article/AuthorList/Author"):
            last = _element_text(author.find("LastName"))
            initials = _element_text(author.find("Initials"))
            if last:
                authors.append(f"{last} {initials}".strip())

        year_node = article.find(".//Article/Journal/JournalIssue/PubDate/Year")
        year_text = (
            _element_text(year_node)
            or _element_text(article.find(".//Article/Journal/JournalIssue/PubDate/MedlineDate"))[
                :4
            ]
        )

        metadata = DocumentMetadata(
            title=title,
            pmid=pmid,
            journal=_element_text(article.find(".//Article/Journal/ISOAbbreviation"))
            or _element_text(article.find(".//Article/Journal/Title")),
            authors=authors,
            publication_year=int(year_text) if year_text.isdigit() else None,
            mesh_terms=[
                _element_text(term)
                for term in article.findall(".//MeshHeadingList/MeshHeading/DescriptorName")
            ],
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
        )

        body_text = normalise_whitespace("\n\n".join([title, *abstract_parts]))
        documents.append(
            Document.create(
                source_type=SourceType.BIOMEDICAL_ABSTRACT,
                text=body_text,
                metadata=metadata,
                sections=sections,
                natural_key=pmid or body_text,
            )
        )
    return documents


def summarise_records(documents: list[Document]) -> dict[str, Any]:
    """Small aggregate used by the acquisition script's console output."""
    years = [d.metadata.publication_year for d in documents if d.metadata.publication_year]
    return {
        "records": len(documents),
        "with_mesh_terms": sum(1 for d in documents if d.metadata.mesh_terms),
        "year_min": min(years) if years else None,
        "year_max": max(years) if years else None,
    }
