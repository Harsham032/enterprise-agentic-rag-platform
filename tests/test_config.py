"""Configuration loading, validation and override behaviour."""

from __future__ import annotations

import pytest

from rag_platform.config import PipelineConfig, Settings
from rag_platform.errors import ConfigurationError


def test_default_config_loads(config: PipelineConfig) -> None:
    assert config.retrieval.top_k > 0
    assert config.embedding.backend == "tfidf_svd"
    assert config.evaluation.primary_k in config.evaluation.k_values


def test_missing_file_raises() -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        PipelineConfig.from_yaml("configs/does-not-exist.yaml")


def test_override_parses_scalars(config: PipelineConfig) -> None:
    updated = config.with_overrides({"retrieval.top_k": "20", "reranking.enabled": "false"})
    assert updated.retrieval.top_k == 20
    assert updated.reranking.enabled is False
    # The original is untouched, so a variant sweep cannot corrupt its base.
    assert config.retrieval.top_k != 20 or config.reranking.enabled is True


def test_unknown_override_key_raises(config: PipelineConfig) -> None:
    with pytest.raises(ConfigurationError, match="unknown configuration key"):
        config.with_overrides({"retrieval.nonexistent": "1"})


def test_overlap_must_be_smaller_than_target(config: PipelineConfig) -> None:
    with pytest.raises(ConfigurationError):
        config.with_overrides({"chunking.overlap_tokens": "400", "chunking.target_tokens": "100"})


def test_candidate_k_must_cover_top_k(config: PipelineConfig) -> None:
    with pytest.raises(ConfigurationError):
        config.with_overrides({"retrieval.candidate_k": "2", "retrieval.top_k": "10"})


def test_primary_k_must_be_in_k_values(config: PipelineConfig) -> None:
    with pytest.raises(ConfigurationError):
        config.with_overrides({"evaluation.primary_k": "7"})


def test_settings_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_API_PORT", "9001")
    monkeypatch.setenv("RAG_ENV", "production")
    settings = Settings(_env_file=None)
    assert settings.api_port == 9001
    assert settings.is_production is True


def test_settings_defaults_carry_no_credentials() -> None:
    settings = Settings(_env_file=None)
    assert settings.openai_api_key == ""
    assert settings.anthropic_api_key == ""
    assert settings.ncbi_api_key == ""


def test_logging_level_can_be_changed_after_first_use(capsys: pytest.CaptureFixture[str]) -> None:
    """A script's --log-level flag must take effect even though modules configure at import."""
    from rag_platform.logging_utils import configure_logging, get_logger

    configure_logging("INFO")
    logger = get_logger("level-test")
    logger.info("visible_at_info")
    assert "visible_at_info" in capsys.readouterr().out

    configure_logging("WARNING")
    logger.info("suppressed_at_warning")
    logger.warning("visible_at_warning")
    captured = capsys.readouterr().out
    assert "suppressed_at_warning" not in captured
    assert "visible_at_warning" in captured

    configure_logging("INFO")
