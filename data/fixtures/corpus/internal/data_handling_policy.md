---
source_type: user_document
title: Document Ingestion and Data Handling Policy
---

## Scope

This policy covers every document ingested into the retrieval platform, whether acquired from a public source, supplied by a customer or generated internally.

## Source Classification

Documents are classified at ingestion into one of three tiers. Public tier covers material published without access restriction, including filings retrieved from the public regulatory archive and citation records retrieved from public bibliographic databases. Licensed tier covers material obtained under an agreement that restricts redistribution. Restricted tier covers material containing personal data or subject to a data use agreement.

## Redistribution

Licensed and restricted tier documents are never redistributed. They are not committed to source control, not included in container images and not returned in full through the query endpoint; only the retrieved passage and its citation are returned. Public tier documents may be cached locally but the acquisition scripts remain the canonical method of obtaining them, so that the upstream terms of use continue to apply.

## Retention

Raw acquired documents are retained for 90 days in the raw storage tier. Derived chunks and embeddings are retained for the life of the index. A deletion request removes the source document, every chunk derived from it and the corresponding embedding rows, and triggers an index rebuild within 24 hours.

## Attribution Requirements

Every answer returned by the platform carries citations resolving to the specific passage used. Answers whose groundedness score falls below the configured threshold are returned with an explicit insufficient-evidence notice rather than a synthesised claim.
