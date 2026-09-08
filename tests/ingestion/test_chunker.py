"""Tests for chunker token overlap logic and Chunk model."""

from unittest.mock import Mock

from app.ingestion.chunker import Chunk, apply_token_overlap


def test_apply_token_overlap_skips_when_chunk_at_cap() -> None:
    # Setup two chunks where chunk 1 is already at max_tokens cap (100 tokens)
    chunks = [
        Chunk(index=0, text="First chunk content", page_start=1, page_end=1, section="", content_type="text", token_count=50),
        Chunk(index=1, text="Second chunk content at cap", page_start=1, page_end=1, section="", content_type="text", token_count=100),
    ]

    docling_tokenizer = Mock()
    # When checking chunk 1's token count, return 100
    docling_tokenizer.count_tokens.side_effect = lambda text: 100 if "Second" in text else 50

    hf_tokenizer = Mock()

    result = apply_token_overlap(
        chunks=chunks,
        overlap_tokens=20,
        max_tokens=100,
        docling_tokenizer=docling_tokenizer,
        hf_tokenizer=hf_tokenizer,
    )

    # Overlap should be skipped for chunk 1 because budget = 100 - 100 = 0 <= 0
    assert result[1].overlap_tokens == 0
    assert result[1].overlap_from_chunk_index is None
    assert result[1].text == "Second chunk content at cap"
    hf_tokenizer.encode.assert_not_called()


def test_apply_token_overlap_caps_to_budget() -> None:
    # Chunk 0 has tokens, Chunk 1 has 85 tokens with max_tokens=100 -> budget is 15 tokens.
    # Requested overlap is 30 tokens. Effective overlap must be capped at 15.
    chunks = [
        Chunk(index=0, text="First chunk content tail", page_start=1, page_end=1, section="", content_type="text", token_count=50),
        Chunk(index=1, text="Second chunk content", page_start=1, page_end=1, section="", content_type="text", token_count=85),
    ]

    docling_tokenizer = Mock()
    docling_tokenizer.count_tokens.side_effect = lambda text: 98 if "tail" in text else (85 if "Second" in text else 50)

    hf_tokenizer = Mock()
    hf_tokenizer.encode.return_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    hf_tokenizer.decode.return_value = "tail"

    result = apply_token_overlap(
        chunks=chunks,
        overlap_tokens=30,
        max_tokens=100,
        docling_tokenizer=docling_tokenizer,
        hf_tokenizer=hf_tokenizer,
    )

    assert result[1].overlap_tokens == 15
    assert result[1].overlap_from_chunk_index == 0
    assert "tail" in result[1].text
    assert result[1].token_count == 98
