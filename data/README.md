# Data directory

## Layout

```
data/
├── fixtures/          committed - the evaluation corpus and its judgements
│   ├── corpus/
│   │   ├── sec/         8 filing-style documents
│   │   ├── biomedical/ 10 structured abstracts
│   │   └── internal/    2 operational documents
│   └── qrels.json     38 labelled queries
├── raw/               ignored - documents as acquired from upstream
├── interim/           ignored - cleaned intermediates
└── processed/         ignored - chunks, indexes, the local SQLite database
```

Only `fixtures/` is committed. Everything else is in `.gitignore`, and CI fails
if a tracked file exceeds 2 MB.

## The evaluation corpus

20 documents, roughly 6,800 words, producing 77 chunks under the default
configuration.

**The documents are fictional.** Companies, studies, journals and every figure
in them are invented. They follow the structure of the real sources - 10-K Item
headings, Background/Methods/Results/Conclusions abstracts, the hedged
disclosure language filings actually use - so that section segmentation, metadata
extraction, chunking and citation labelling are exercised on realistic input.

They exist so the published benchmark is reproducible offline, by anyone, with
no credentials and no rate-limited download. See `docs/data-sources.md` for why
fictional entities were chosen over real ones, and `docs/results.md` section 2
for what it means for interpreting the results.

### Document format

Markdown with YAML front matter:

```markdown
---
source_type: sec_filing
company_name: Northwind Energy Corporation
ticker: NWE
cik: '1000101'
form_type: 10-K
fiscal_year: 2024
filing_date: '2025-02-18'
---

## Item 1A. Risk Factors

...
```

`source_type` is one of `sec_filing`, `biomedical_abstract`, `user_document`.
Every other key maps onto `DocumentMetadata`; unknown keys are preserved. Files
without front matter load fine and have their metadata inferred.

### Judgement format

```json
{
  "query_id": "q001",
  "query": "Which turbine supplier concentration risk does Northwind Energy disclose?",
  "type": "single_hop",
  "relevant": [
    {"document": "northwind_energy_10k_fy2024.md", "sections": ["Item 1A. Risk Factors"]}
  ],
  "answer_terms": ["two manufacturers", "84 percent", "alternative supplier"]
}
```

`relevant` names document regions rather than chunk ids, so judgements stay
valid when the chunking configuration changes. `answer_terms` are the content
elements a complete answer must contain; they are used only for the completeness
metric and never for retrieval.

## Acquiring real data

Both upstreams require an identifying contact address and enforce a request-rate
ceiling. Set these in `.env` first:

```bash
RAG_SEC_USER_AGENT="Your Name your.email@example.com"
RAG_NCBI_EMAIL="your.email@example.com"
```

Then:

```bash
python scripts/download_data.py sec --ticker AAPL --forms 10-K --limit 2 --facts
python scripts/download_data.py pubmed --term "sickle cell gene therapy" --limit 50
python scripts/preprocess.py --input data/raw --output data/processed/chunks.jsonl
python scripts/build_index.py --corpus data/raw
```

### Approximate storage

| Item | Size |
| --- | --- |
| One 10-K primary document (HTML) | 2-8 MB |
| The same after text extraction | 300-800 KB |
| XBRL company facts for a large filer | 5-30 MB JSON |
| One PubMed record | 2-4 KB |
| 1,000 chunks with 256-dimension embeddings | roughly 1 MB |

### Licensing

Read `docs/data-sources.md` before acquiring anything. In short: SEC filings are
public-domain US government works; PubMed abstract copyright generally rests
with the publisher, so only titles, abstracts and metadata are retrieved and
none of it is redistributed here.
