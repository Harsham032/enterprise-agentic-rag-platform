# Data sources

Terms, access method and redistribution position for every source the platform
reads. No source data beyond the purpose-written evaluation corpus is committed
to this repository.

## SEC EDGAR

| | |
| --- | --- |
| Publisher | United States Securities and Exchange Commission |
| Content used | Form 10-K, 10-Q and 8-K primary documents; company submission history; XBRL company facts |
| Access | Public HTTP, no key or registration |
| Endpoints | `data.sec.gov/submissions/`, `data.sec.gov/api/xbrl/companyfacts/`, `www.sec.gov/Archives/edgar/data/`, `www.sec.gov/files/company_tickers.json` |
| Terms | https://www.sec.gov/os/webmaster-faq#developers |
| Rate limit | 10 requests per second, published |
| Identification | A `User-Agent` header carrying a contact email address is required |
| Redistribution | Filings are US government works in the public domain. This repository still does not commit them: the acquisition script stays the canonical way to obtain them, so the upstream terms continue to apply. |

The client enforces 8 requests per second, below the published ceiling, and
refuses to start without a contact address in the user agent.

```bash
export RAG_SEC_USER_AGENT="Your Name your.email@example.com"
python scripts/download_data.py sec --ticker AAPL --forms 10-K --limit 2 --facts
```

Expected volume: a single 10-K primary document is typically 2-8 MB of HTML,
reducing to roughly 300-800 KB of extracted text. Company facts for a large
filer run to 5-30 MB of JSON.

## PubMed / NCBI E-utilities

| | |
| --- | --- |
| Publisher | National Center for Biotechnology Information, US National Library of Medicine |
| Content used | Titles, abstracts, bibliographic metadata, MeSH terms |
| Access | Public HTTP; an optional API key raises the rate limit |
| Endpoints | `eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi`, `.../efetch.fcgi` |
| Terms | https://www.ncbi.nlm.nih.gov/books/NBK25497/ |
| Rate limit | 3 requests per second without a key, 10 with one |
| Identification | `tool` and `email` parameters are required on every request |
| Redistribution | **Abstract copyright generally rests with the publisher, not with NCBI.** Records may be retrieved and used, but this repository redistributes none of them. |

Only titles, abstracts and metadata are retrieved. Full text is deliberately not
fetched: most of it sits behind publisher licences that forbid redistribution,
and the abstract carries the retrievable signal for this task.

```bash
export RAG_NCBI_EMAIL="your.email@example.com"
python scripts/download_data.py pubmed --term "sickle cell gene therapy" --limit 50
```

Expected volume: roughly 2-4 KB per record.

## User-supplied documents

| | |
| --- | --- |
| Content used | PDF, HTML, Markdown and plain text |
| Access | Placed in a corpus directory, or posted to `POST /documents` |
| Terms | Whatever the supplier's own terms are |
| Redistribution | None. Uploaded documents are not committed, not included in container images, and only the retrieved passage is returned through the query endpoint. |

Scanned PDFs without an embedded text layer yield an empty extraction rather
than silently producing garbage. Route those through OCR before ingestion.

## The bundled evaluation corpus

`data/fixtures/corpus` holds 20 purpose-written documents used by the benchmark
and by the test suite.

**These documents are fictional.** The companies, studies, journals, figures and
identifiers in them are invented. They are written to the structure of real
filings and structured abstracts so the pipeline is exercised on realistic
input, but nothing in them describes a real entity or a real result.

The alternative would have been to attribute invented financial figures to real
companies or invented trial results to real journals, which would be worse in
every respect. `docs/results.md` section 2 states what this means for how the
benchmark numbers should be read.

## Handling rules

These apply to every source and are enforced by `.gitignore` and by the
`quality` CI workflow:

1. Acquired documents land in `data/raw/`, which is ignored by git.
2. Derived artefacts land in `data/interim/` and `data/processed/`, also ignored.
3. No file over 2 MB may be tracked; CI fails the build if one is.
4. `.env` is never committed; `.env.example` carries placeholders only.
5. Credentials are read from the environment and never appear in configuration
   files committed to the repository.
