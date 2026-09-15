"""Embedding backends, first-stage retrievers, fusion and reranking."""

from __future__ import annotations

import numpy as np
import pytest

from rag_platform.config import EmbeddingConfig, RerankingConfig, RetrievalConfig
from rag_platform.data.models import Chunk, ScoredChunk, SourceType
from rag_platform.errors import EmbeddingError, IndexNotBuiltError
from rag_platform.models.embeddings import TfidfSvdEmbedder, build_embedder, l2_normalise
from rag_platform.models.reranker import (
    FEATURE_NAMES,
    IdentityReranker,
    LogisticReranker,
    build_reranker,
    extract_features,
)
from rag_platform.retrieval.dense import DenseRetriever
from rag_platform.retrieval.hybrid import (
    HybridRetriever,
    min_max_normalise,
    reciprocal_rank_fusion,
    weighted_fusion,
)
from rag_platform.retrieval.lexical import BM25Retriever

CORPUS = [
    "Total revenue increased by 2.8 percent driven by services growth",
    "Net income declined due to higher operating expenses and tax",
    "The clinical trial enrolled 45 patients with sickle cell disease",
    "Fetal haemoglobin levels rose after gene editing therapy",
    "Supply chain concentration in a single region is a material risk",
]


def _chunks(texts: list[str]) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"c{i}",
            doc_id=f"d{i}",
            source_type=SourceType.SEC_FILING if i < 2 else SourceType.BIOMEDICAL_ABSTRACT,
            text=text,
            position=i,
            char_start=0,
            char_end=len(text),
        )
        for i, text in enumerate(texts)
    ]


# --------------------------------------------------------------- embeddings


def test_l2_normalise_handles_zero_rows() -> None:
    matrix = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    normalised = l2_normalise(matrix)
    assert normalised[0] @ normalised[0] == pytest.approx(1.0, abs=1e-5)
    assert np.all(normalised[1] == 0.0)


def test_tfidf_svd_produces_unit_vectors() -> None:
    embedder = build_embedder(EmbeddingConfig(dim=4, min_df=1, max_df=1.0))
    embedder.fit(CORPUS)
    matrix = embedder.encode(CORPUS)
    assert matrix.shape == (5, embedder.dim)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-4)


def test_tfidf_svd_ranks_a_paraphrase_highest() -> None:
    embedder = build_embedder(EmbeddingConfig(dim=4, min_df=1, max_df=1.0))
    embedder.fit(CORPUS)
    similarities = (
        embedder.encode(CORPUS) @ embedder.encode(["gene editing therapy for sickle cell"])[0]
    )
    assert int(np.argmax(similarities)) in {2, 3}


def test_encode_before_fit_raises() -> None:
    with pytest.raises(EmbeddingError, match="fit must be called"):
        TfidfSvdEmbedder(EmbeddingConfig(dim=4)).encode(["anything"])


def test_fit_on_empty_corpus_raises() -> None:
    with pytest.raises(EmbeddingError, match="empty corpus"):
        TfidfSvdEmbedder(EmbeddingConfig(dim=4)).fit([])


def test_requested_dimension_is_capped_by_the_corpus() -> None:
    embedder = TfidfSvdEmbedder(EmbeddingConfig(dim=512, min_df=1, max_df=1.0))
    embedder.fit(CORPUS)
    assert embedder.dim < 512
    assert 0.0 < embedder.explained_variance <= 1.0


def test_embedder_round_trips_through_disk(tmp_path) -> None:
    embedder = TfidfSvdEmbedder(EmbeddingConfig(dim=4, min_df=1, max_df=1.0))
    embedder.fit(CORPUS)
    before = embedder.encode(["revenue growth"])
    path = tmp_path / "embedder.joblib"
    embedder.save(path)

    restored = TfidfSvdEmbedder(EmbeddingConfig(dim=4, min_df=1, max_df=1.0))
    restored.load(path)
    assert np.allclose(before, restored.encode(["revenue growth"]))


def test_unknown_backend_raises() -> None:
    config = EmbeddingConfig(dim=4)
    config.backend = "nonexistent"  # type: ignore[assignment]
    with pytest.raises(EmbeddingError, match="unknown embedding backend"):
        build_embedder(config)


# ------------------------------------------------------------------ lexical


def test_bm25_finds_the_exact_term_match() -> None:
    retriever = BM25Retriever()
    retriever.build(_chunks(CORPUS))
    assert retriever.search("sickle cell disease patients", 3)[0][0] == 2


def test_bm25_stemming_matches_inflected_forms() -> None:
    stemmed = BM25Retriever(stemming=True)
    stemmed.build(_chunks(CORPUS))
    assert stemmed.search("enrolling patient", 3)

    unstemmed = BM25Retriever(stemming=False)
    unstemmed.build(_chunks(CORPUS))
    assert not unstemmed.search("enrolling patient", 3)


def test_bm25_scoring_before_build_raises() -> None:
    with pytest.raises(IndexNotBuiltError):
        BM25Retriever().scores("anything")


def test_bm25_build_on_empty_corpus_raises() -> None:
    with pytest.raises(IndexNotBuiltError):
        BM25Retriever().build([])


def test_bm25_stopword_only_query_returns_nothing() -> None:
    retriever = BM25Retriever()
    retriever.build(_chunks(CORPUS))
    assert retriever.search("the and of", 3) == []


# -------------------------------------------------------------------- dense


def test_dense_retriever_returns_ranked_cosines() -> None:
    embedder = build_embedder(EmbeddingConfig(dim=4, min_df=1, max_df=1.0))
    retriever = DenseRetriever(embedder)
    retriever.build(_chunks(CORPUS))
    results = retriever.search("fetal haemoglobin gene editing", 3)
    assert len(results) == 3
    assert results[0][1] >= results[-1][1]


def test_dense_scoring_before_build_raises() -> None:
    with pytest.raises(IndexNotBuiltError):
        DenseRetriever(build_embedder(EmbeddingConfig(dim=4))).scores("anything")


# ------------------------------------------------------------------- fusion


def test_min_max_normalise_maps_constant_input_to_zero() -> None:
    assert np.all(min_max_normalise(np.array([2.0, 2.0, 2.0])) == 0.0)


def test_weighted_fusion_respects_the_weights() -> None:
    lexical = {0: 10.0, 1: 0.0}
    dense = {0: 0.0, 1: 1.0}
    lexical_heavy = weighted_fusion(lexical, dense, lexical_weight=1.0, dense_weight=0.0)
    dense_heavy = weighted_fusion(lexical, dense, lexical_weight=0.0, dense_weight=1.0)
    assert lexical_heavy[0] > lexical_heavy[1]
    assert dense_heavy[1] > dense_heavy[0]


def test_weighted_fusion_on_empty_input() -> None:
    assert weighted_fusion({}, {}, lexical_weight=0.5, dense_weight=0.5) == {}


def test_rrf_rewards_agreement_between_retrievers() -> None:
    fused = reciprocal_rank_fusion([[7, 1, 2], [3, 7, 4]], k=60)
    # Item 7 is ranked highly by both lists, so it must beat either list's top item.
    assert max(fused, key=lambda key: fused[key]) == 7


def test_hybrid_retriever_filters_by_source_type() -> None:
    config = RetrievalConfig(top_k=3, candidate_k=5)
    hybrid = HybridRetriever(
        BM25Retriever(),
        DenseRetriever(build_embedder(EmbeddingConfig(dim=4, min_df=1, max_df=1.0))),
        config,
    )
    hybrid.build(_chunks(CORPUS))
    results = hybrid.search("revenue and patients", source_types={SourceType.SEC_FILING.value})
    assert results
    assert all(result.chunk.source_type is SourceType.SEC_FILING for result in results)


def test_hybrid_search_before_build_raises() -> None:
    config = RetrievalConfig()
    hybrid = HybridRetriever(
        BM25Retriever(), DenseRetriever(build_embedder(EmbeddingConfig(dim=4))), config
    )
    with pytest.raises(IndexNotBuiltError):
        hybrid.search("anything")


# ---------------------------------------------------------------- reranking


def _candidate(text: str, score: float, rank: int, heading: str | None = None) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=f"c{rank}",
        doc_id="d",
        source_type=SourceType.SEC_FILING,
        text=text,
        position=rank,
        section_heading=heading,
        char_start=0,
        char_end=len(text),
    )
    return ScoredChunk(chunk=chunk, score=score, lexical_score=score, dense_score=score, rank=rank)


def test_feature_vector_matches_declared_names() -> None:
    features = extract_features(
        "turbine supplier risk", _candidate("turbine suppliers are concentrated", 0.9, 1)
    )
    assert features.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(features))


def test_heading_match_feature_fires() -> None:
    index = FEATURE_NAMES.index("heading_match")
    with_heading = extract_features(
        "risk factors", _candidate("body", 0.5, 1, "Item 1A. Risk Factors")
    )
    without = extract_features("risk factors", _candidate("body", 0.5, 1))
    assert with_heading[index] > without[index]


def test_exact_phrase_feature_fires() -> None:
    index = FEATURE_NAMES.index("exact_phrase")
    assert (
        extract_features(
            "supply concentration", _candidate("our supply concentration is high", 0.5, 1)
        )[index]
        == 1.0
    )
    assert (
        extract_features("supply concentration", _candidate("unrelated text entirely", 0.5, 1))[
            index
        ]
        == 0.0
    )


def test_unfitted_reranker_falls_back_to_first_stage_order() -> None:
    reranker = LogisticReranker()
    candidates = [_candidate("a", 0.2, 1), _candidate("b", 0.9, 2)]
    assert [c.chunk_id for c in reranker.rerank("q", candidates, 2)] == ["c2", "c1"]


def test_reranker_learns_from_labelled_examples() -> None:
    reranker = LogisticReranker()
    examples = []
    for i in range(12):
        relevant = i % 2 == 0
        text = (
            "turbine supplier concentration risk" if relevant else "unrelated warehouse throughput"
        )
        examples.append(
            (
                "turbine supplier risk",
                _candidate(text, 0.9 if relevant else 0.1, i + 1),
                int(relevant),
            )
        )
    coefficients = reranker.fit(examples)
    assert set(coefficients) == set(FEATURE_NAMES)
    assert reranker.is_fitted

    ranked = reranker.rerank(
        "turbine supplier risk",
        [
            _candidate("unrelated warehouse throughput", 0.5, 1),
            _candidate("turbine supplier concentration risk", 0.5, 2),
        ],
        2,
    )
    assert "turbine" in ranked[0].chunk.text
    assert ranked[0].rerank_score is not None


def test_reranker_ignores_single_class_training_data() -> None:
    reranker = LogisticReranker()
    assert reranker.fit([("q", _candidate("a", 0.5, 1), 0)]) == {}
    assert not reranker.is_fitted


def test_reranker_round_trips_through_disk(tmp_path) -> None:
    reranker = LogisticReranker()
    examples = [
        ("q", _candidate("turbine supplier concentration", 0.9, i + 1), i % 2) for i in range(10)
    ]
    reranker.fit(examples)
    path = tmp_path / "reranker.joblib"
    reranker.save(path)

    restored = LogisticReranker()
    restored.load(path)
    assert restored.is_fitted


def test_saving_an_unfitted_reranker_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="unfitted"):
        LogisticReranker().save(tmp_path / "x.joblib")


def test_disabled_reranking_builds_a_passthrough() -> None:
    assert isinstance(build_reranker(RerankingConfig(enabled=False)), IdentityReranker)
    assert isinstance(build_reranker(RerankingConfig(model="none")), IdentityReranker)
