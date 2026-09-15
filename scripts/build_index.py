#!/usr/bin/env python
"""Build the retrieval index, persist metadata and fit the serving reranker.

Run this before starting the service so the reranker used in production is the
one the benchmark measured. Without a fitted reranker the serving path falls
back to the first-stage fused score.

Usage::

    python scripts/build_index.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_platform.config import PipelineConfig, load_settings
from rag_platform.data.local_documents import load_directory
from rag_platform.data.store import MetadataStore
from rag_platform.evaluation.qrels import build_span_index, load_qrels
from rag_platform.logging_utils import configure_logging, get_logger
from rag_platform.retrieval.index import RetrievalIndex
from rag_platform.utils.seeds import set_global_seed

logger = get_logger("build_index")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--corpus", default=None, help="override the corpus directory")
    parser.add_argument("--output-dir", default=None, help="where to write the reranker artifact")
    parser.add_argument("--skip-reranker", action="store_true", help="do not fit the reranker")
    parser.add_argument("--skip-database", action="store_true", help="do not write metadata rows")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    config = PipelineConfig.from_yaml(args.config)
    if args.corpus:
        config = config.with_overrides({"corpus.source_dir": args.corpus})
    set_global_seed(config.run.seed)

    documents = load_directory(Path(config.corpus.source_dir))
    index = RetrievalIndex(config)
    index.build_from_documents(documents)

    output_dir = Path(args.output_dir or config.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {"index": index.describe()}

    if not args.skip_database:
        settings = load_settings()
        store = MetadataStore(settings.database_url)
        store.create_all()
        store.upsert_documents(index.documents)
        store.upsert_chunks(index.chunks)
        summary["database"] = {
            "url": settings.database_url,
            "documents": store.document_count(),
            "chunks": store.chunk_count(),
        }
        store.close()

    if not args.skip_reranker and config.reranking.enabled and config.corpus.qrels_path:
        qrels = load_qrels(config.corpus.qrels_path)
        # Fit on the training split only. Fitting on every judged query would
        # make the offline benchmark and the serving model disagree, and would
        # leave no honest way to report what reranking is worth.
        train_ids, _ = qrels.training_split(0.5, seed=config.run.seed)
        spans = build_span_index(index.documents)
        training = {
            judgement.query: judgement.relevant_chunk_ids(index.chunks, spans)
            for judgement in qrels.queries
            if judgement.query_id in set(train_ids)
        }
        coefficients = index.fit_reranker(training)
        if coefficients:
            artifact = output_dir / "reranker.joblib"
            index.save_reranker(artifact)
            summary["reranker"] = {
                "artifact": str(artifact),
                "training_queries": len(training),
                "coefficients": coefficients,
            }

    (output_dir / "index_build.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
