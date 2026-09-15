# Experimental results

Every number on this page was produced by `scripts/run_experiment.py` from a
clean checkout and can be reproduced with one command. Nothing here is an
estimate, a projection, or a figure carried over from another system.

```bash
make install
python scripts/run_experiment.py --sweep --output-dir reports
```

The machine-readable report, including per-query results, is written to
`reports/experiment_report.json`.

## 1. Experiment environment

| | |
| --- | --- |
| Platform | Linux 6.18 x86_64, glibc 2.39 |
| CPU | x86_64, 4 cores |
| Python | 3.11.15 |
| NumPy | 2.4.6 |
| scikit-learn | 1.9.1 |
| Package version | 0.1.0 |
| Seed | 20260101 |
| Run date | 2026-09-15 |
| Test suite at time of run | 250 tests, all passing |

Latency figures are wall-clock on this machine with no accelerator. They are
useful for comparing configurations against each other and should not be read
as a capacity estimate for any other hardware.

## 2. Dataset

The benchmark runs on the evaluation corpus bundled at `data/fixtures/corpus`.

| | |
| --- | --- |
| Documents | 20 |
| Words | approximately 6,800 |
| Chunks (default configuration) | 77 |
| Mean chunk length | 76.5 whitespace tokens |
| Chunks by source | 28 filing, 40 biomedical, 9 internal |
| Labelled queries | 38 (32 single-hop, 6 comparison) |
| Judged regions | 46 document/section targets |

**What this corpus is, stated plainly.** It is a purpose-written evaluation set,
not a sample of real filings or real abstracts. The companies, studies and
figures in it are invented. Two reasons drove that choice:

1. *Reproducibility.* The metrics on this page have to be reproducible by
   anyone who clones the repository, offline, with no credentials. A corpus that
   must be downloaded at run time from a rate-limited upstream is not that.
2. *Honesty.* Attributing invented financial figures to real companies, or
   invented trial results to real journals, would be worse than any alternative.
   Inventing the entities as well keeps the documents clearly fictional.

The documents are written to the structure of the real sources - 10-K Item
headings, structured abstract sections, the kind of hedged disclosure language
filings actually use - so that section-aware chunking, metadata extraction and
citation labelling are exercised on realistic input.

**What follows from that.** These numbers characterise this pipeline on this
corpus. They are not a claim about performance on SEC EDGAR or PubMed at scale,
and nothing here has been measured on live data from either source. Section 7
sets out what would have to be run to make such a claim.

## 3. Data split

Judged queries are split 50/50 by query id, deterministically from the seed.

* **Training split, 19 queries.** Used to fit the reranker and to select the
  chunking configuration.
* **Held-out split, 19 queries.** Every number in section 5 comes from this
  split, for every configuration including the ones that use no learned
  component. Scoring the reranker on queries it was fitted on would have
  produced a better-looking and meaningless number.

Splitting by query rather than by candidate matters: the reranker sees roughly
thirty candidates per query, so a candidate-level split would leak every
training query into the evaluation set.

## 4. Metrics

**Relevance is defined over document regions, not chunks.** A judgement names a
document and a set of section headings; those resolve to character ranges, and a
chunk counts as relevant when either 40 percent of the chunk falls inside the
judged region or the chunk contains 60 percent of it. Requiring only the first
test would penalise a chunker for carrying context; requiring only the second
would reward a chunker that returns whole documents. This is what makes one set
of labels valid across every chunking configuration compared below.

**`recall@budget` is the headline metric.** It truncates the ranking at a fixed
evidence budget of 400 whitespace tokens and measures recall over what fits.
This is the only retrieval metric on this page that is comparable across chunk
sizes: `recall@5` returns three times as much text at 250-token chunks as at
75-token chunks, so comparing configurations at fixed *k* compares two different
things. Section 6 shows how badly that goes wrong.

Answer quality is measured lexically, with no judge model:

* **groundedness** - share of answer sentences whose content words appear in
  some retrieved passage;
* **citation supported** - share of citations whose quote is present in the
  chunk they point at;
* **answer completeness** - share of the content elements a complete answer
  needs (a figure, an entity, a direction of change), listed per query in the
  judgement file.

`docs/methodology.md` states what these measures cannot detect.

## 5. Configuration comparison

Held-out split, 19 queries, 400-token evidence budget. `recall@budget` carries a
95 percent bootstrap interval over queries.

| Configuration | Recall@budget | Recall@5 | NDCG@5 | MRR | Answer completeness | p95 latency (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| `lexical_only` | 0.675 (0.465-0.860) | 0.807 | 0.648 | 0.632 | 0.557 | 31.7 |
| `dense_only` | 0.667 (0.447-0.851) | 0.772 | 0.661 | 0.708 | 0.588 | 31.5 |
| `hybrid_rrf` | 0.693 (0.491-0.868) | 0.798 | 0.643 | 0.648 | 0.583 | 31.5 |
| `hybrid_weighted` | 0.754 (0.553-0.921) | 0.807 | 0.651 | 0.660 | 0.610 | 32.4 |
| `hybrid_reranked` **(default)** | **0.772** (0.605-0.921) | **0.851** | **0.760** | **0.791** | **0.623** | 50.5 |
| `hybrid_reranked_no_stemming` | 0.772 (0.605-0.921) | 0.798 | 0.713 | 0.749 | 0.588 | 48.2 |
| `fixed_window_chunking` | 0.544 (0.385-0.706) | 0.544 | 0.516 | 0.677 | 0.588 | 24.9 |

The retrieval columns measure a single retrieval call. The answer-completeness
and latency columns measure the full request path, including query
decomposition and the entity scoping described below - which is why a change
that leaves retrieval untouched can still move completeness.

### Interpretation

**Read the intervals before the point estimates.** With 19 queries the bootstrap
intervals span roughly 0.3 and overlap heavily. The ordering below is what the
data suggests; only the chunking result is separated clearly enough to be called
a finding. Everything else is directional.

* **Hybrid fusion beats either retriever alone.** 0.754 for weighted fusion
  against 0.675 lexical and 0.667 dense. Lexical retrieval wins on queries
  naming an exact identifier (a fiscal year, a gene, a us-gaap concept); latent
  semantic indexing wins on paraphrase. The intervals overlap, so treat this as
  consistent with the usual result rather than as independent evidence for it.

* **Weighted fusion beats reciprocal rank fusion here** (0.754 against 0.693).
  RRF discards score magnitude, which costs it on a corpus where BM25 scores are
  well separated. On a corpus with a heavier BM25 tail the robustness RRF buys
  would likely be worth more.

* **Reranking helps ordering more than it helps recall.** Recall@budget moves
  0.754 to 0.772, well inside the noise. NDCG@5 moves 0.651 to 0.760 and MRR
  0.660 to 0.791, which is the larger and more consistent effect: the reranker
  is promoting the right chunk within an already-good candidate set rather than
  finding chunks fusion missed. That is what a reranker is supposed to do. It
  costs roughly 27ms at p95.

  The learned weights are written to `reports/index_build.json` by
  `make index` and are inspectable there. The largest
  are the fused first-stage score (3.43), chunk length (2.71) and the dense
  score (2.18); `exact_phrase` gets a coefficient of 0.0 because no training
  query contained a phrase appearing verbatim in a chunk.

* **Stemming changes ranking, not recall.** Recall@budget is identical at 0.772,
  but NDCG@5 is 0.760 with stemming against 0.713 without, MRR 0.791 against
  0.749, and answer completeness 0.623 against 0.588. Inflectional matching
  ("turbine"/"turbines") affects which of the retrieved chunks ends up first and
  which sentences the synthesiser selects.

* **Section-aware chunking beats fixed windows decisively at matched
  granularity**: 0.772 against 0.544, with intervals that barely overlap. This
  is the one clearly separated result on the page, and section 6 explains why
  the naive version of this comparison says the opposite.

### Results by query type

`hybrid_reranked`, held-out split:

| Query type | Queries | Recall@budget | Recall@5 | NDCG@5 | MRR | Mean relevant chunks |
| --- | --- | --- | --- | --- | --- | --- |
| single-hop | 16 | 0.875 | 0.938 | 0.834 | 0.836 | 1.5 |
| comparison | 3 | 0.222 | 0.389 | 0.364 | 0.548 | 3.7 |

**This is the system's clearest weakness and the aggregate hides it.** Comparison
queries score 0.222 against 0.875 for single-hop. These figures are for a single
retrieval call, without decomposition; the answer-level effect after
decomposition is smaller but still clear (completeness 0.472 against 0.651). Two
things contribute:

1. A comparison query has 3.7 relevant chunks on average against 1.5 for a
   single-hop query, so roughly 280 tokens of genuinely relevant material
   competes for a 400-token budget alongside everything else retrieved.
2. Decomposition divides the budget across sub-queries, so each period or entity
   gets a fraction of the evidence a single-hop query would get.

Only three comparison queries are in the held-out split, so 0.222 is a very
imprecise estimate - but the direction is consistent with the training split and
with the mechanism, and it should not be averaged away. Section 7 lists what
would address it.

### Entity scoping: a correctness bug, not a ranking problem

An early version of this pipeline answered *"Compare Northwind Energy revenue
growth in 2024 against 2023"* by quoting Atlas Payments' revenue figures, with a
groundedness score of 1.000. Every sentence was faithfully copied from a
retrieved chunk; the chunks were about the wrong company.

The cause is structural rather than a tuning problem. Decomposition turns that
query into "Northwind Energy revenue growth in 2024" and "Northwind Energy
revenue growth in 2023". Those two sub-queries share almost every content word
with the corresponding sub-query about any other company in the corpus, so term
and vector scores cannot reliably separate them - the discriminating tokens are
two words out of seven.

The fix: the planner records which entity each sub-query is about, and retrieval
applies it as a metadata filter over company name, title and ticker. Matching on
document metadata rather than chunk body text matters, because a filing
routinely names a competitor inside its own risk factors. The filter is dropped
when it matches nothing, so an unrecognised name degrades to an unscoped search
rather than to an empty result.

Measured effect on the held-out split: answer completeness rose from 0.596 to
0.623 overall and from 0.583 to 0.610 for `hybrid_weighted`. Retrieval metrics
are unchanged, because they measure a single retrieval call that never had an
entity to scope by.

**The interesting part is what the metrics did not catch.** Groundedness was
1.000 both before and after. A lexical faithfulness measure cannot tell the
difference between a correct quotation and a correct quotation from the wrong
document, which is exactly the limitation recorded in section 7 and in
`docs/methodology.md`. The bug was found by reading an example answer, not by
reading a metric. `tests/test_generation.py::test_comparison_evidence_stays_within_the_named_entities`
guards it now.

### Answer quality

`hybrid_reranked`, held-out split, extractive generation:

| Measure | Value | 95 percent interval |
| --- | --- | --- |
| Groundedness | 1.000 | 1.000-1.000 |
| Citations resolvable | 1.000 | - |
| Citations supported | 1.000 | 1.000-1.000 |
| Answer completeness | 0.623 | 0.439-0.789 |
| Mean citations per answer | 2.89 | - |

By query type: completeness 0.651 on the 16 single-hop queries, 0.472 on the 3
comparison queries.

**Groundedness of 1.000 is a design guarantee, not an achievement.** The
extractive synthesiser copies sentences verbatim out of retrieved chunks, so
every sentence is supported by construction and any other value would be a bug.
The same applies to citation support. These are reported because a regression in
either would indicate a real defect in the citation plumbing, not because they
say the system is unusually faithful.

**Answer completeness of 0.623 is the number that actually measures answer
quality**, and it is the honest one: a little under two thirds of the content
elements a complete answer needs are present. The interval is wide. The failure
mode behind the misses is visible in the per-query results: the synthesiser
selects a span containing the entity but not the figure, usually when the figure
sits two or more sentences after the vocabulary the query matched.

### Latency

`hybrid_reranked`, held-out split, single process, no accelerator:

| Stage | p50 | p95 |
| --- | --- | --- |
| Retrieval only | 8.1ms | 8.9ms |
| End to end (plan, retrieve, synthesise) | 32.9ms | 50.5ms |

Index build over 20 documents takes 87ms, of which most is fitting the TF-IDF
and SVD projection. On a corpus of this size an exact dense scan is faster than
an approximate index and loses no recall.

## 6. Chunk size is confounded with chunking strategy

Running the strategy comparison naively produced a result that looked clear and
was wrong. At each strategy's own default target, fixed-window chunking appeared
to beat section-aware chunking on every fixed-*k* metric. It does not. The
larger chunks a fixed window produced were being credited with more of the
judged material simply for being longer.

Chunk-size sweep, held-out split:

| Configuration | Chunks | Mean chunk tokens | Recall@budget | Precision@budget | Chunks per budget | Recall@5 |
| --- | --- | --- | --- | --- | --- | --- |
| `section_aware_400` | 77 | 76.5 | **0.754** | 0.284 | 4.8 | 0.807 |
| `section_aware_300` | 77 | 76.5 | **0.754** | 0.284 | 4.8 | 0.807 |
| `section_aware_220` | 83 | 73.2 | 0.605 | 0.275 | 4.9 | 0.675 |
| `section_aware_180` | 84 | 72.5 | 0.623 | 0.276 | 5.1 | 0.675 |
| `section_aware_120` | 92 | 67.9 | 0.601 | 0.295 | 5.3 | 0.601 |
| `section_aware_80` | 111 | 59.9 | 0.535 | 0.250 | 6.3 | 0.516 |
| `fixed_window_400` | 26 | 248.3 | 0.728 | 0.684 | 1.2 | 0.982 |
| `fixed_window_300` | 29 | 226.8 | 0.684 | 0.789 | 1.2 | 0.947 |
| `fixed_window_220` | 37 | 186.4 | 0.570 | 0.500 | 1.4 | 0.829 |
| `fixed_window_180` | 51 | 146.2 | 0.605 | 0.553 | 2.2 | 0.737 |
| `fixed_window_120` | 70 | 110.1 | 0.535 | 0.333 | 3.1 | 0.583 |
| `fixed_window_80` | 101 | 77.4 | 0.538 | 0.337 | 5.0 | 0.538 |

Three things to take from this table.

**The `recall@5` column is not a fair comparison and is included to show why.**
`fixed_window_400` reaches 0.982 - but the corpus is only 26 chunks at that
setting, so returning five of them means returning 19 percent of the entire
corpus. Push chunk size far enough and every chunk becomes a document; recall@5
approaches 1.0 and means nothing. `recall@budget` corrects this to 0.728 because
only 1.2 of those large chunks fit inside 400 tokens.

**At matched chunk length, section-aware chunking wins clearly.** Compare
`section_aware_400` at 76.5 mean tokens with `fixed_window_80` at 77.4: 0.754
against 0.538. Sections are coherent units of meaning, so a section-sized chunk
carries a complete risk factor or a complete trial result, while a window of the
same length cuts across the boundary between two.

**The target size does not bind for section-aware chunking on this corpus.** At
300 and 400 tokens the results are identical because no section in the corpus
exceeds 300 tokens: the chunker emits one chunk per section. The default is set
to 300 to make that explicit. On a corpus of full 10-K filings, where Item 1A
regularly runs to thousands of words, the target would bind and the sweep would
need re-running.

**A note on how the default was chosen.** The chunking configuration was
selected on the training split, not on the held-out split above. Selecting it by
`ndcg@budget` alone would have picked `fixed_window_400` - the degenerate
26-chunk configuration - because NDCG normalises by the ideal ranking at *k*,
and with only 1.2 chunks per budget a single relevant chunk scores near 1.0.
`recall@budget` was used instead, which does not have that failure mode.

## 7. Limitations

Stated as flatly as possible.

1. **The corpus is small and synthetic.** 20 documents, 77 chunks, roughly 6,800
   words. Retrieval difficulty rises sharply with corpus size and with the
   near-duplicate documents a real filing archive contains. Nothing here
   predicts behaviour at 100,000 documents.

2. **Nineteen held-out queries cannot separate close configurations.** The
   bootstrap intervals span about 0.3. Only the chunking-strategy result is
   separated clearly enough to be called a finding.

3. **The default embedding backend is latent semantic indexing, not a pretrained
   encoder.** TF-IDF plus SVD fit on the corpus has no world knowledge and
   cannot match vocabulary it has never seen. A pretrained bi-encoder would very
   likely raise dense retrieval, and the backend is wired for it - but it needs a
   model download, so it is not what the reproducible benchmark measures. No
   number on this page should be read as a statement about that backend.

4. **Answer quality is measured lexically.** Groundedness and completeness are
   token-overlap measures. They cannot detect a claim that is faithfully
   paraphrased but wrong, and they under-credit a correct answer phrased in
   vocabulary the evidence does not use. They are a conservative lower bound on
   faithfulness, not a substitute for human adjudication.

5. **Abstractive generation is unmeasured.** Everything above uses the
   extractive path. The model-backed synthesiser is implemented and unit-tested
   against fakes; it has not been benchmarked, because doing so needs an API key
   and would make the results non-reproducible for anyone without one.

6. **No live acquisition was exercised.** The SEC and PubMed clients are
   tested against recorded upstream payloads covering request construction,
   rate limiting, retry behaviour and parsing. They have not been run against
   the live services in the environment these results were produced in, where
   outbound access to `sec.gov` and `eutils.ncbi.nlm.nih.gov` is blocked by
   policy. Both fail with an explicit, actionable error in that case.

7. **Comparison queries are weak and only three of them are held out.** The
   0.222 figure is directionally real but very imprecisely estimated.

8. **The container image was not built in the environment that produced these
   results** (no Docker daemon available). The Dockerfile and the compose stack
   are exercised by the `quality` CI workflow, which builds the image, starts it
   and asserts that `/health` reports `ok`.

## 8. Next experiments

In the order that would most change what is known:

1. **Run the same benchmark on live SEC and PubMed data.** The acquisition
   scripts already produce corpora in the format the harness reads. This is the
   single change that would turn these results into a claim about the real
   sources, and it needs only network access plus labelling effort.

2. **Enlarge the judged query set to 150-200 queries.** At the current size the
   intervals swamp most of the differences. This is the cheapest way to make the
   configuration comparison decisive.

3. **Benchmark the pretrained bi-encoder backend against latent semantic
   indexing** on the same split, reporting the latency cost alongside the
   quality gain. Limitation 3 is currently an assumption.

4. **Keep fixing comparison queries.** Entity scoping addressed cross-document
   contamination; the remaining gap is budget allocation. Two candidate changes,
   to be measured separately: allocate the evidence budget per sub-query rather
   than dividing a fixed *k*, and have the synthesiser select one span per
   sub-query rather than selecting globally across merged evidence.

5. **Measure the abstractive path** on groundedness and completeness against the
   extractive baseline, and quantify what it costs in latency and in citations
   that fail verification.

6. **Add near-duplicate documents to the corpus.** Consecutive annual filings
   from one company are close to duplicates in the sections that matter, and
   distinguishing them is a large share of the real difficulty. The corpus has
   two such pairs; a realistic archive has dozens.
