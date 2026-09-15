#!/usr/bin/env python
"""Clean and chunk an acquired corpus, writing the result to ``data/processed``.

Separating preprocessing from index building makes chunking changes cheap to
inspect: the output is plain JSONL, so a diff shows exactly how a configuration
change reshaped the corpus.

Usage::

    python scripts/preprocess.py --input data/raw --output data/processed/chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_platform.config import PipelineConfig
from rag_platform.data.local_documents import load_directory
from rag_platform.features.chunking import chunk_documents
from rag_platform.features.cleaning import clean_document_text, drop_boilerplate_lines
from rag_platform.features.metadata import enrich_metadata
from rag_platform.logging_utils import configure_logging, get_logger

logger = get_logger("preprocess")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--input", default=None, help="corpus directory; defaults to the configured one"
    )
    parser.add_argument("--output", default="data/processed/chunks.jsonl")
    parser.add_argument("--no-clean", action="store_true", help="skip boilerplate removal")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    config = PipelineConfig.from_yaml(args.config)
    source_dir = Path(args.input or config.corpus.source_dir)

    documents = load_directory(source_dir)
    if not args.no_clean:
        documents = [
            document.model_copy(
                update={"text": drop_boilerplate_lines(clean_document_text(document.text))}
            )
            for document in documents
        ]
    documents = [enrich_metadata(document) for document in documents]
    chunks = chunk_documents(documents, config.chunking)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.model_dump(mode="json")) + "\n")

    by_source = Counter(chunk.source_type.value for chunk in chunks)
    token_counts = [chunk.token_count for chunk in chunks]
    logger.info(
        "preprocess_complete",
        documents=len(documents),
        chunks=len(chunks),
        output=str(output),
        mean_tokens=round(sum(token_counts) / len(token_counts), 1) if token_counts else 0,
        by_source=dict(by_source),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
