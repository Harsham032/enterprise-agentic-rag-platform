"""Parsing, cleaning, chunking and metadata enrichment."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_platform.config import ChunkingConfig
from rag_platform.data.local_documents import load_document, split_front_matter
from rag_platform.data.models import Document, DocumentMetadata, Section, SourceType
from rag_platform.errors import ParsingError
from rag_platform.features.chunking import chunk_document, chunk_documents
from rag_platform.features.cleaning import clean_document_text, drop_boilerplate_lines
from rag_platform.features.metadata import enrich_metadata, infer_fiscal_year, infer_form_type
from rag_platform.features.parsing import extract_text, parse_html, parse_markdown

# ------------------------------------------------------------------ parsing


def test_parse_html_removes_scripts_and_separates_blocks() -> None:
    html = (
        "<html><head><style>p{color:red}</style></head><body><p>Alpha</p><p>Beta</p></body></html>"
    )
    assert parse_html(html) == "Alpha\nBeta"


def test_parse_markdown_returns_sections() -> None:
    text, sections = parse_markdown("# One\nbody one\n\n## Two\nbody two")
    assert [section.heading for section in sections] == ["One", "Two"]
    assert "body one" in text


def test_parse_markdown_without_headings_returns_no_sections() -> None:
    _, sections = parse_markdown("just a paragraph with no heading")
    assert sections == []


def test_extract_text_rejects_unsupported_suffix(tmp_path: Path) -> None:
    path = tmp_path / "data.parquet"
    path.write_bytes(b"\x00")
    with pytest.raises(ParsingError, match="unsupported"):
        extract_text(path)


# ----------------------------------------------------------------- cleaning


def test_clean_removes_page_furniture_and_dot_leaders() -> None:
    raw = "Table of Contents\nItem 1 ......... 4\nPage 3 of 10\nReal content here."
    cleaned = clean_document_text(raw)
    assert "Table of Contents" not in cleaned
    assert "Page 3 of 10" not in cleaned
    assert "Real content here." in cleaned


def test_drop_boilerplate_removes_repeated_short_lines() -> None:
    text = "HDR\na real line of text\nHDR\nmore text here\nHDR"
    assert drop_boilerplate_lines(text) == "a real line of text\nmore text here"


def test_drop_boilerplate_keeps_long_repeated_lines() -> None:
    long_line = " ".join(["word"] * 20)
    text = f"{long_line}\nother\n{long_line}\nother two\n{long_line}"
    assert long_line in drop_boilerplate_lines(text)


# ----------------------------------------------------------------- chunking


def test_section_aware_chunking_carries_headings(sample_document: Document) -> None:
    chunks = chunk_document(
        sample_document, ChunkingConfig(target_tokens=40, overlap_tokens=10, min_tokens=5)
    )
    assert chunks
    assert {chunk.section_heading for chunk in chunks} == {
        "Item 1A. Risk Factors",
        "Item 7. Management's Discussion and Analysis",
    }


def test_fixed_window_chunking_respects_target(sample_document: Document) -> None:
    config = ChunkingConfig(
        strategy="fixed_window", target_tokens=20, overlap_tokens=5, min_tokens=5
    )
    chunks = chunk_document(sample_document, config)
    assert all(chunk.token_count <= 20 for chunk in chunks)
    assert len(chunks) > 1


def test_fixed_window_chunks_overlap(sample_document: Document) -> None:
    config = ChunkingConfig(
        strategy="fixed_window", target_tokens=20, overlap_tokens=8, min_tokens=5
    )
    chunks = chunk_document(sample_document, config)
    first_tail = set(chunks[0].text.split()[-8:])
    second_head = set(chunks[1].text.split()[:8])
    assert first_tail & second_head


@pytest.mark.parametrize("strategy", ["section_aware", "fixed_window"])
def test_chunk_offsets_locate_text_in_the_document(
    sample_document: Document, strategy: str
) -> None:
    """Offsets must be usable coordinates: relevance judgements depend on them."""
    config = ChunkingConfig(strategy=strategy, target_tokens=25, overlap_tokens=5, min_tokens=5)
    for chunk in chunk_document(sample_document, config):
        assert 0 <= chunk.char_start < chunk.char_end <= len(sample_document.text) + 1
        head = chunk.text.split()[0]
        assert head in sample_document.text[chunk.char_start : chunk.char_end + len(head)]


def test_chunk_ids_are_stable_across_runs(sample_document: Document) -> None:
    config = ChunkingConfig(target_tokens=30, overlap_tokens=5, min_tokens=5)
    first = [chunk.chunk_id for chunk in chunk_document(sample_document, config)]
    second = [chunk.chunk_id for chunk in chunk_document(sample_document, config)]
    assert first == second


def test_short_document_yields_one_chunk() -> None:
    document = Document.create(
        source_type=SourceType.USER_DOCUMENT, text="Two words.", natural_key="tiny"
    )
    chunks = chunk_document(
        document, ChunkingConfig(target_tokens=50, overlap_tokens=5, min_tokens=40)
    )
    assert len(chunks) == 1


def test_empty_document_yields_no_chunks() -> None:
    document = Document.create(source_type=SourceType.USER_DOCUMENT, text="", natural_key="empty")
    assert chunk_document(document, ChunkingConfig()) == []


def test_oversized_sentence_is_not_split_mid_clause() -> None:
    sentence = " ".join(["token"] * 120) + "."
    document = Document.create(
        source_type=SourceType.USER_DOCUMENT,
        text=sentence,
        sections=[Section(heading="One", text=sentence, order=0)],
        natural_key="long",
    )
    chunks = chunk_document(
        document, ChunkingConfig(target_tokens=40, overlap_tokens=10, min_tokens=5)
    )
    assert len(chunks) == 1


def test_chunk_documents_processes_a_corpus(corpus: list[Document], config) -> None:
    chunks = chunk_documents(corpus, config.chunking)
    assert len(chunks) > len(corpus)
    assert all(chunk.doc_id for chunk in chunks)


# ----------------------------------------------------------------- metadata


def test_infer_fiscal_year_from_cover_page() -> None:
    assert infer_fiscal_year("For the fiscal year ended December 31, 2024") == 2024


def test_infer_form_type() -> None:
    assert infer_form_type("ANNUAL REPORT ON FORM 10-K") == "10-K"


def test_enrich_does_not_overwrite_supplied_metadata(sample_document: Document) -> None:
    enriched = enrich_metadata(sample_document)
    assert enriched.metadata.fiscal_year == 2024
    assert enriched.metadata.title == "Example Filing"


def test_enrich_fills_missing_fields() -> None:
    document = Document.create(
        source_type=SourceType.SEC_FILING,
        text="Annual report on Form 10-Q for the fiscal year ended June 30, 2022. Details follow.",
        metadata=DocumentMetadata(),
        natural_key="bare",
    )
    enriched = enrich_metadata(document)
    assert enriched.metadata.fiscal_year == 2022
    assert enriched.metadata.form_type == "10-Q"
    assert enriched.metadata.title


# ------------------------------------------------------------- front matter


def test_split_front_matter_parses_yaml() -> None:
    metadata, body = split_front_matter(
        "---\nsource_type: sec_filing\nfiscal_year: 2024\n---\n\nBody text"
    )
    assert metadata == {"source_type": "sec_filing", "fiscal_year": 2024}
    assert body.strip() == "Body text"


def test_split_front_matter_passes_through_plain_text() -> None:
    metadata, body = split_front_matter("No front matter here")
    assert metadata == {}
    assert body == "No front matter here"


def test_malformed_front_matter_raises() -> None:
    with pytest.raises(ParsingError, match="front matter"):
        split_front_matter("---\ntitle: a: b: c\n\tbad\n---\n\nbody")


def test_unknown_source_type_raises(tmp_path: Path) -> None:
    path = tmp_path / "doc.md"
    path.write_text("---\nsource_type: mystery\n---\n\n# Title\nbody", encoding="utf-8")
    with pytest.raises(ParsingError, match="unknown source_type"):
        load_document(path)


def test_load_document_records_source_file(tmp_path: Path) -> None:
    path = tmp_path / "policy_note.md"
    path.write_text("# Heading\n\nSome body text for the note.", encoding="utf-8")
    document = load_document(path)
    assert document.metadata.source_file == "policy_note.md"
    assert document.source_type is SourceType.USER_DOCUMENT
