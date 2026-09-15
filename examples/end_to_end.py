#!/usr/bin/env python
"""End-to-end walkthrough: build an index, answer questions, score the result.

Run it from the repository root with no arguments:

    python examples/end_to_end.py

It uses the bundled evaluation corpus and needs no network access, no
credentials and no external service.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_platform.agents.orchestrator import Orchestrator
from rag_platform.config import PipelineConfig
from rag_platform.data.local_documents import load_directory
from rag_platform.evaluation.harness import evaluate_retrieval
from rag_platform.evaluation.qrels import load_qrels
from rag_platform.logging_utils import configure_logging
from rag_platform.retrieval.index import RetrievalIndex
from rag_platform.utils.seeds import set_global_seed

QUESTIONS = [
    "Which turbine supplier concentration risk does Northwind Energy disclose?",
    "Compare Northwind Energy revenue growth in 2024 against 2023",
    "What fetal haemoglobin level was achieved after BCL11A enhancer editing?",
    "Which studies report that a model degraded when evaluated outside its development setting?",
]


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main() -> int:
    # Quiet: the point of the example is the output, not the log stream.
    configure_logging("WARNING")

    root = Path(__file__).resolve().parents[1]
    config = PipelineConfig.from_yaml(root / "configs" / "default.yaml")
    set_global_seed(config.run.seed)

    rule("Building the index")
    documents = load_directory(root / config.corpus.source_dir)
    index = RetrievalIndex(config)
    index.build_from_documents(documents)
    stats = index.describe()
    print(
        f"{stats['documents']} documents, {stats['chunks']} chunks, "
        f"mean {stats['mean_chunk_tokens']:.1f} tokens per chunk, "
        f"built in {stats['build_ms']:.0f}ms"
    )

    # The reranker is fitted offline by scripts/build_index.py. Without it the
    # serving path falls back to the first-stage fused score.
    artifact = root / config.run.output_dir / "reranker.joblib"
    print("reranker:", "loaded" if index.load_reranker(artifact) else "not fitted (run make index)")

    orchestrator = Orchestrator(index, config)

    for question in QUESTIONS:
        rule(question)
        answer = orchestrator.answer(question, top_k=6)
        print(answer.text)
        print()
        for citation in answer.citations:
            mark = "verified" if citation.verified else "UNVERIFIED"
            print(f"  [{citation.marker}] {mark} ({citation.support_score:.2f}) {citation.label}")
        print(
            f"\n  strategy={answer.diagnostics['strategy']} "
            f"sub-queries={len(answer.subqueries)} "
            f"groundedness={answer.groundedness:.2f} "
            f"latency={answer.latency_ms:.1f}ms"
        )

    rule("Retrieval quality over the full judged query set")
    qrels = load_qrels(root / config.corpus.qrels_path)
    metrics = evaluate_retrieval(index, qrels)["metrics"]
    for name in ("recall@budget", "recall@5", "precision@5", "ndcg@5", "reciprocal_rank"):
        print(f"  {name:<18} {metrics[name]:.3f}")
    print(
        "\n  Note: this scores all 38 judged queries, including the 19 the reranker"
        "\n  was fitted on. The held-out numbers in docs/results.md are the ones to"
        "\n  compare configurations by."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
