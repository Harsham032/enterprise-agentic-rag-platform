#!/usr/bin/env python
"""Acquire documents from SEC EDGAR and PubMed into ``data/raw``.

Both sources are public and require no API key, but both require the client to
identify itself and to respect a published request-rate ceiling. Set
``RAG_SEC_USER_AGENT`` and ``RAG_NCBI_EMAIL`` in ``.env`` before running.

Nothing downloaded by this script is committed to the repository. See
``data/README.md`` for the terms that apply to each source.

Examples::

    python scripts/download_data.py sec --ticker AAPL --forms 10-K --limit 2
    python scripts/download_data.py pubmed --term "sickle cell gene therapy" --limit 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_platform.config import load_settings
from rag_platform.data.models import Document
from rag_platform.data.pubmed import PubMedClient, summarise_records
from rag_platform.data.sec_edgar import SecEdgarClient, extract_us_gaap_facts, normalise_cik
from rag_platform.errors import DataAcquisitionError
from rag_platform.features.parsing import parse_html
from rag_platform.logging_utils import configure_logging, get_logger

logger = get_logger("download")

DEFAULT_RAW_DIR = Path("data/raw")


def write_document(document: Document, directory: Path) -> Path:
    """Write a document as markdown with YAML front matter."""
    import yaml

    directory.mkdir(parents=True, exist_ok=True)
    front_matter = {
        "source_type": document.source_type.value,
        **{
            key: value
            for key, value in document.metadata.model_dump(mode="json").items()
            if value not in (None, "", [], {})
        },
    }
    body = (
        "\n\n".join(f"## {section.heading}\n\n{section.text}" for section in document.sections)
        or document.text
    )
    path = directory / f"{document.doc_id}.md"
    path.write_text(
        "---\n"
        + yaml.safe_dump(front_matter, sort_keys=False, allow_unicode=True, width=1000)
        + "---\n\n"
        + body
        + "\n",
        encoding="utf-8",
    )
    return path


def download_sec(args: argparse.Namespace) -> int:
    settings = load_settings()
    out_dir = Path(args.output) / "sec"
    with SecEdgarClient(settings.sec_user_agent) as client:
        cik = normalise_cik(args.cik) if args.cik else client.ticker_to_cik(args.ticker)
        filings = client.list_filings(cik, form_types=tuple(args.forms), limit=args.limit)
        if not filings:
            logger.warning("no_filings_found", cik=cik, forms=args.forms)
            return 1

        for filing in filings:
            filing["document_url"] = client.filing_document_url(filing)
            raw = client.fetch_filing_html(filing)
            text = parse_html(raw) if raw.lstrip().startswith("<") else raw
            document = SecEdgarClient.build_document(filing, text)
            path = write_document(document, out_dir)
            logger.info(
                "filing_saved",
                accession=filing["accession_number"],
                form=filing["form"],
                sections=len(document.sections),
                path=str(path),
            )

        if args.facts:
            facts = extract_us_gaap_facts(client.company_facts(cik))
            facts_path = out_dir / f"facts_{cik}.json"
            facts_path.write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")
            logger.info("facts_saved", rows=len(facts), path=str(facts_path))
    return 0


def download_pubmed(args: argparse.Namespace) -> int:
    settings = load_settings()
    out_dir = Path(args.output) / "pubmed"
    with PubMedClient(
        tool=settings.ncbi_tool_name, email=settings.ncbi_email, api_key=settings.ncbi_api_key
    ) as client:
        documents = client.search_and_fetch(args.term, retmax=args.limit)
        if not documents:
            logger.warning("no_records_found", term=args.term)
            return 1
        for document in documents:
            write_document(document, out_dir)
        logger.info("pubmed_saved", directory=str(out_dir), **summarise_records(documents))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", default=str(DEFAULT_RAW_DIR), help="raw data directory")
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="source", required=True)

    sec = subparsers.add_parser("sec", help="download filings from SEC EDGAR")
    identifier = sec.add_mutually_exclusive_group(required=True)
    identifier.add_argument(
        "--ticker", help="exchange ticker, resolved through the EDGAR ticker map"
    )
    identifier.add_argument("--cik", help="central index key")
    sec.add_argument("--forms", nargs="+", default=["10-K"], help="form types to fetch")
    sec.add_argument("--limit", type=int, default=2, help="maximum filings to fetch")
    sec.add_argument("--facts", action="store_true", help="also fetch XBRL company facts")
    sec.set_defaults(handler=download_sec)

    pubmed = subparsers.add_parser("pubmed", help="download records from PubMed")
    pubmed.add_argument("--term", required=True, help="PubMed search expression")
    pubmed.add_argument("--limit", type=int, default=50, help="maximum records to fetch")
    pubmed.set_defaults(handler=download_pubmed)

    args = parser.parse_args()
    configure_logging(args.log_level)
    try:
        return int(args.handler(args))
    except DataAcquisitionError as exc:
        logger.error("acquisition_failed", error=str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
