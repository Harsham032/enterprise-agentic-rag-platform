# Methodology

How the numbers in `docs/results.md` are produced, and what they can and cannot
support.

## 1. Relevance judgements

`data/fixtures/qrels.json` holds 38 queries. Each names the document regions
that answer it and the content elements a complete answer must contain.

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

**Regions, not chunk ids.** Chunk ids are content-derived, so a judgement
written against them would be invalidated by any change to chunk size - which
would make the chunking comparison impossible to run. Judgements therefore name
a document and section headings, which resolve at evaluation time to character
ranges in the document text.

**Resolution rule.** A chunk counts as relevant when either:

* 40 percent or more of the chunk falls inside the judged region - the chunk is
  mostly about the right thing; or
* the chunk contains 60 percent or more of the judged region - the chunk carries
  the answer, even though it also carries surrounding material.

Either test alone breaks the comparison in one direction. Only the first would
penalise a coarse chunker for including context. Only the second would reward a
chunker that returns whole documents. Where a judgement names several adjacent
sections, overlap is measured against their union, so a chunk straddling two
relevant sections is not rejected for failing the threshold on each separately.

**A judgement that resolves to nothing is treated as a corrupt label file**, not
as a hard query. Silently scoring zero on an unresolvable judgement deflates
every aggregate; the harness raises instead.

**Authorship.** The queries were written against the corpus by one author, which
is a known limitation: they reflect one person's idea of what is askable, and
they were not validated by a second annotator. The set is small enough to read
end to end, which is the main defence available at this size.

## 2. Splitting

Queries are split 50/50 by query id, shuffled deterministically from the seed.
The training split fits the reranker and selects the chunking configuration; the
held-out split produces every number reported.

Splitting by query, not by candidate, is essential. The reranker sees around
thirty candidates per query, so a candidate-level split would place candidates
from the same query on both sides and leak the evaluation set into training.

**Every configuration is scored on the same held-out split**, including the ones
with no learned component. Scoring the unreranked configurations on all 38
queries while scoring the reranked one on 19 would not be a comparison.

## 3. Metric definitions

Names are used inconsistently across the retrieval literature, so the exact
definitions are:

| Metric | Definition |
| --- | --- |
| `recall@k` | Judged-relevant chunks in the top *k*, divided by the total judged relevant. Not capped at *k*: when a query has more relevant chunks than *k*, recall cannot reach 1.0, and that ceiling is real. |
| `precision@k` | Relevant chunks in the top *k*, divided by *k* - not by the number returned, which would flatter a short result list. |
| `reciprocal_rank` | One over the rank of the first relevant chunk; zero if none. |
| `ndcg@k` | Binary-gain NDCG with the standard `1/log2(rank+1)` discount, normalised by the ideal ranking for that query. |
| `map@k` | Mean of the precision measured at each relevant hit, divided by the number relevant. |
| `recall@budget` | Recall over the ranking truncated at a fixed number of evidence tokens. |

### Why `recall@budget` is the headline metric

Fixed-*k* metrics are not comparable across chunk sizes. Returning five chunks
of 250 tokens returns 1,250 tokens of evidence; returning five chunks of 75
tokens returns 375. Comparing those at "top 5" compares two different amounts of
text and rewards whichever configuration returns more, independent of ranking
quality.

`recall@budget` truncates each ranking at 400 whitespace tokens - the constraint
a real system faces, a bounded context window - and measures recall over what
fits. At least one chunk is always included, so a single chunk larger than the
budget is still scored.

The degenerate case this prevents is worth stating: push chunk size up far
enough and every chunk becomes a document, `recall@5` approaches 1.0, and the
metric is measuring nothing. `docs/results.md` section 6 shows this happening
(`fixed_window_400` reaching `recall@5` 0.982 while returning 19 percent of the
corpus, corrected to 0.728 under a budget).

**`ndcg@budget` is computed but is not used for selection.** Because the number
of chunks fitting the budget varies by configuration, and NDCG normalises by the
ideal ranking at that *k*, a configuration where only one chunk fits scores near
1.0 whenever that chunk is relevant. Selecting on it picks the most degenerate
configuration available, which is exactly what happened before the failure mode
was identified.

## 4. Uncertainty

Every headline metric carries a 95 percent percentile bootstrap interval over
queries: 2,000 resamples with replacement, seeded.

With 19 held-out queries the intervals span roughly 0.3. **A configuration whose
interval overlaps another's has not been shown to beat it.** In the reported
results only the chunking-strategy comparison is separated clearly enough to be
called a finding; everything else is directional.

This is reported prominently rather than in a footnote because the alternative -
quoting point estimates from 19 queries as though they were settled - is the
most common way a retrieval benchmark misleads.

## 5. Answer quality

Three lexical measures, computed with no judge model.

**Groundedness.** The share of answer sentences whose content words appear in
some retrieved passage, at a containment threshold of 0.35. Sentences with fewer
than three content words are skipped rather than counted as unsupported, since
containment says nothing useful about them.

**Citation correctness.** Two quantities, separated because their causes differ:
*resolvable* (the cited chunk id is in the result set - a failure is a plumbing
bug) and *supported* (the quote is present in the cited chunk - a failure means
the citation points at the wrong passage).

**Answer completeness.** The share of `answer_terms` present in the answer.
Multi-word terms match as substrings of the normalised answer; single tokens
match against the token set, so `0.71` does not match `0.718`.

### What these measures cannot do

Stated plainly, because lexical faithfulness metrics are routinely over-read:

* They cannot detect a claim that is faithfully paraphrased but factually wrong.
* They under-credit a correct answer phrased in vocabulary the evidence does not
  use, which penalises abstractive generation relative to extractive.
* `answer_terms` encode one author's view of what a complete answer contains.

They are a conservative lower bound on faithfulness, not a substitute for human
adjudication. On the extractive path groundedness is 1.0 by construction, which
makes it useful as a regression detector and useless as a quality claim.

## 6. Reproducibility

* All randomness derives from one seed (`run.seed`, default 20260101), applied
  to Python's RNG, NumPy, the SVD solver, the reranker's solver, the query split
  and the bootstrap.
* The corpus and judgements are committed, so nothing is downloaded at
  evaluation time.
* The benchmark runs in CI on every change, and the report is uploaded as an
  artefact, so the published numbers cannot silently drift from the code.
* `tests/test_pipeline.py::test_index_build_is_deterministic` asserts that two
  builds of the same corpus produce identical rankings.

Reproducing from a clean checkout:

```bash
make install
python scripts/run_experiment.py --sweep --output-dir reports
```

## 7. Threats to validity

| Threat | Effect | Mitigation, or why not |
| --- | --- | --- |
| Small corpus (20 documents) | Retrieval is easier than at realistic scale | None. Stated as a limitation; the fix is running on live data. |
| Small query set (19 held out) | Intervals swamp most differences | Bootstrap intervals reported; no claim made where they overlap. |
| Single annotator | Judgements reflect one view of relevance | None available at this size. |
| Synthetic corpus | May not reflect real document messiness | Documents follow real structural conventions; the parsers are separately tested against recorded real-format payloads. |
| Configuration selected on labelled data | Risk of overfitting the corpus | Selection done on the training split only; held-out numbers reported. |
| Lexical answer metrics | Cannot detect paraphrased errors | Stated; extractive generation makes the specific failure they miss impossible. |
| Author wrote both corpus and queries | Queries may be unrealistically well matched to the text | Partly unavoidable. Queries deliberately use vocabulary that differs from the documents in several cases; the resulting misses are visible in the per-query results. |
