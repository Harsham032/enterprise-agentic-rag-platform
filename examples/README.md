# Examples

## `end_to_end.py`

Builds the index from the bundled corpus, answers four questions covering each
planning strategy, and reports retrieval metrics. No network, no credentials, no
external service.

```bash
python examples/end_to_end.py
```

## `sample_query_output.json`

A real `POST /query` response for three query shapes — a single-hop lookup, a
temporal comparison and a cross-source question — captured from the pipeline at
the configuration reported in `docs/results.md`. Evidence is truncated to the
first two chunks per query for readability; nothing else is edited.

Worth noticing in it:

- Citation labels resolve to a document, a form type, a fiscal year and a
  section, not to an opaque chunk id.
- Every citation carries the quote it was derived from and a `verified` flag set
  by checking that quote against the chunk it points at.
- The comparison query's `subqueries` field shows the decomposition, and all of
  its citations come from the same company — the entity scoping described in
  `docs/results.md`.
- `diagnostics` breaks latency into planning, retrieval and generation, which is
  what makes a slow query diagnosable.
