#!/usr/bin/env python
"""Run the retrieval and generation benchmark and write a reproducible report.

Evaluates a set of configurations on a held-out query split and writes both a
machine-readable report and a markdown summary. Every configuration is scored on
the same split, so the comparison between them is like for like.

Usage::

    python scripts/run_experiment.py --config configs/default.yaml
    python scripts/run_experiment.py --only baseline_lexical hybrid_reranked
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_platform import __version__
from rag_platform.config import PipelineConfig
from rag_platform.data.local_documents import load_directory
from rag_platform.evaluation.harness import run_configuration
from rag_platform.evaluation.qrels import load_qrels
from rag_platform.logging_utils import configure_logging, get_logger
from rag_platform.utils.seeds import set_global_seed

logger = get_logger("experiment")


def configuration_variants(base: PipelineConfig) -> dict[str, PipelineConfig]:
    """The configurations compared by the benchmark.

    Each isolates one decision so the report attributes any difference to that
    decision rather than to a bundle of changes.
    """
    return {
        # First stage only, no fusion: what a pure keyword system achieves.
        "lexical_only": base.with_overrides(
            {
                "retrieval.lexical_weight": 1.0,
                "retrieval.dense_weight": 0.0,
                "reranking.enabled": False,
            }
        ),
        # Latent semantic indexing only.
        "dense_only": base.with_overrides(
            {
                "retrieval.lexical_weight": 0.0,
                "retrieval.dense_weight": 1.0,
                "reranking.enabled": False,
            }
        ),
        # Weighted fusion of both, no reranking.
        "hybrid_weighted": base.with_overrides({"reranking.enabled": False}),
        # Rank-based fusion of both, no reranking.
        "hybrid_rrf": base.with_overrides({"retrieval.fusion": "rrf", "reranking.enabled": False}),
        # Fusion plus the learned reranker: the default serving configuration.
        "hybrid_reranked": base,
        # Ablation: fusion plus reranking with stemming switched off.
        "hybrid_reranked_no_stemming": base.with_overrides({"retrieval.stemming": False}),
        # Ablation: fixed-window chunking at a comparable mean chunk length.
        # Comparing it at its own best chunk size instead would compare two
        # things at once; see chunking_sweep below for the size dimension.
        "fixed_window_chunking": base.with_overrides(
            {
                "chunking.strategy": "fixed_window",
                "chunking.target_tokens": 80,
                "chunking.overlap_tokens": 20,
            }
        ),
    }


def chunking_sweep_variants(base: PipelineConfig) -> dict[str, PipelineConfig]:
    """Chunk-size sweep across both strategies.

    Reported separately because chunk size and chunking strategy are confounded:
    a larger chunk carries more text, so it is credited with more of the judged
    material whatever strategy produced it. Only the budgeted metrics, which
    equalise the number of evidence tokens returned, are comparable down this
    table.
    """
    variants: dict[str, PipelineConfig] = {}
    for strategy in ("section_aware", "fixed_window"):
        for target, overlap in ((400, 40), (300, 40), (220, 40), (180, 40), (120, 30), (80, 20)):
            variants[f"{strategy}_{target}"] = base.with_overrides(
                {
                    "chunking.strategy": strategy,
                    "chunking.target_tokens": target,
                    "chunking.overlap_tokens": overlap,
                    "reranking.enabled": False,
                }
            )
    return variants


def environment() -> dict[str, Any]:
    """Facts about the machine and versions a reader needs to interpret timings."""
    import numpy
    import sklearn

    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "scikit_learn": sklearn.__version__,
        "package_version": __version__,
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def markdown_table(results: list[dict[str, Any]], primary_k: int) -> str:
    """Render the headline comparison as a markdown table."""
    header = (
        f"| Configuration | Recall@budget | Recall@{primary_k} | NDCG@{primary_k} | MRR "
        f"| Answer completeness | p95 latency (ms) |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
    )
    rows = []
    for result in results:
        metrics = result["retrieval"]["metrics"]
        answers = result.get("answers", {}).get("metrics", {})
        latency = result.get("answers", {}).get("latency", result["retrieval"]["latency"])
        interval = result["retrieval"]["confidence_intervals"].get("recall@budget", {})
        budget_recall = metrics.get("recall@budget", 0.0)
        budget_cell = (
            f"**{budget_recall:.3f}** ({interval['lower']:.3f}-{interval['upper']:.3f})"
            if interval
            else f"{budget_recall:.3f}"
        )
        completeness = answers.get("answer_completeness")
        rows.append(
            f"| `{result['name']}` | {budget_cell} | {metrics.get(f'recall@{primary_k}', 0.0):.3f} "
            f"| {metrics.get(f'ndcg@{primary_k}', 0.0):.3f} | {metrics.get('reciprocal_rank', 0.0):.3f} "
            f"| {'-' if completeness is None else f'{completeness:.3f}'} "
            f"| {latency.get('p95_ms', 0.0):.1f} |"
        )
    return header + "\n".join(rows) + "\n"


def sweep_table(results: list[dict[str, Any]]) -> str:
    """Render the chunk-size sweep, including the quantities that confound it."""
    header = (
        "| Configuration | Chunks | Mean chunk tokens | Recall@budget | Precision@budget "
        "| Chunks per budget | Recall@5 |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
    )
    rows = []
    for result in results:
        metrics = result["retrieval"]["metrics"]
        index = result["index"]
        rows.append(
            f"| `{result['name']}` | {index['chunks']} | {index['mean_chunk_tokens']:.1f} "
            f"| {metrics.get('recall@budget', 0.0):.3f} | {metrics.get('precision@budget', 0.0):.3f} "
            f"| {metrics.get('budget_chunks', 0.0):.1f} | {metrics.get('recall@5', 0.0):.3f} |"
        )
    return header + "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml", help="pipeline configuration")
    parser.add_argument("--corpus", default=None, help="override the corpus directory")
    parser.add_argument("--qrels", default=None, help="override the judgement file")
    parser.add_argument("--output-dir", default=None, help="where to write the report")
    parser.add_argument("--only", nargs="*", default=None, help="run only these configurations")
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.5,
        help="share of queries used to fit the reranker",
    )
    parser.add_argument("--skip-answers", action="store_true", help="retrieval metrics only")
    parser.add_argument("--sweep", action="store_true", help="also run the chunk-size sweep")
    parser.add_argument(
        "--split",
        choices=["test", "train"],
        default="test",
        help="which query split to report on; 'train' is for configuration selection only",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    base = PipelineConfig.from_yaml(args.config)
    if args.corpus:
        base = base.with_overrides({"corpus.source_dir": args.corpus})
    if args.qrels:
        base = base.with_overrides({"corpus.qrels_path": args.qrels})
    set_global_seed(base.run.seed)

    documents = load_directory(Path(base.corpus.source_dir))
    qrels = load_qrels(base.corpus.qrels_path)
    train_ids, test_ids = qrels.training_split(args.train_fraction, seed=base.run.seed)
    logger.info(
        "experiment_start",
        documents=len(documents),
        queries=len(qrels),
        train=len(train_ids),
        test=len(test_ids),
    )

    variants = configuration_variants(base)
    if args.only:
        missing = [name for name in args.only if name not in variants]
        if missing:
            parser.error(f"unknown configuration(s): {', '.join(missing)}")
        variants = {name: variants[name] for name in args.only}

    report_ids = train_ids if args.split == "train" else test_ids
    results = [
        run_configuration(
            name,
            config,
            documents,
            qrels,
            train_ids,
            report_ids,
            include_answers=not args.skip_answers,
        )
        for name, config in variants.items()
    ]

    sweep_results: list[dict[str, Any]] = []
    if args.sweep:
        sweep_results = [
            run_configuration(
                name, config, documents, qrels, train_ids, report_ids, include_answers=False
            )
            for name, config in chunking_sweep_variants(base).items()
        ]

    output_dir = Path(args.output_dir or base.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "environment": environment(),
        "corpus": {
            "source_dir": str(base.corpus.source_dir),
            "documents": len(documents),
            "queries_total": len(qrels),
            "queries_train": len(train_ids),
            "queries_test": len(test_ids),
            "train_query_ids": list(train_ids),
            "test_query_ids": list(test_ids),
        },
        "seed": base.run.seed,
        "reported_split": args.split,
        "token_budget": base.evaluation.token_budget,
        "results": results,
        "chunking_sweep": sweep_results,
    }
    report_path = output_dir / "experiment_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")

    table = markdown_table(results, base.evaluation.primary_k)
    summary = (
        f"# Benchmark summary\n\n"
        f"Reported split: `{args.split}` ({len(report_ids)} of {len(qrels)} queries). "
        f"Evidence budget: {base.evaluation.token_budget} tokens. "
        f"Recall@budget shown with a 95 percent bootstrap interval.\n\n{table}"
    )
    if sweep_results:
        summary += f"\n## Chunk-size sweep\n\n{sweep_table(sweep_results)}"
    (output_dir / "experiment_summary.md").write_text(summary, encoding="utf-8")

    print(
        f"\nReported split: {args.split} ({len(report_ids)} queries); training split: {len(train_ids)}\n"
    )
    print(table)
    if sweep_results:
        print(sweep_table(sweep_results))
    print(f"Report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
