---
source_type: user_document
title: Retrieval Service Operations Runbook
---

## Service Level Objectives

The query endpoint targets a ninety-fifth percentile end-to-end latency of 400 milliseconds for single-hop queries against an index of up to 250,000 chunks, and 1,200 milliseconds for decomposed multi-hop queries. Availability target is 99.5 percent measured monthly, excluding scheduled index rebuild windows.

## Index Rebuild Procedure

Index rebuilds are required whenever the embedding backend changes, whenever the chunking configuration changes, or when more than 10 percent of the corpus has been added since the last rebuild. Rebuilds run against a shadow index; traffic is cut over only after the evaluation harness reports recall at 5 within two percentage points of the incumbent index on the standing query set. A rebuild that fails this gate is rolled back automatically and does not page.

## Alerting

Three alerts page the on-call engineer. First, groundedness below 0.80 averaged over a rolling one-hour window, which usually indicates that retrieval is returning off-topic evidence rather than that generation has regressed. Second, ninety-fifth percentile query latency above 1,500 milliseconds for ten consecutive minutes. Third, a zero-result rate above 5 percent, which most often means the index failed to load after a deployment.

## Known Failure Modes

The most common production failure is a stale embedding backend: the index is rebuilt but the serving process keeps the previous fitted vectoriser in memory, so query vectors land in a different space from document vectors. The symptom is a collapse in dense scores with lexical scores unaffected, which the diagnostics block in the query response exposes directly. Restarting the serving process resolves it.
