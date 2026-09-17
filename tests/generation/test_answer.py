"""Tests for answer generation and citation parsing."""

from unittest.mock import Mock
import pytest

from app.generation.answer import (
    NO_INFO_MESSAGE,
    generate_grounded_answer,
    parse_citations,
)
from app.retrieval.models import ChunkSearchResult


def test_parse_citations_extracts_bracketed_indices() -> None:
    text = "Revenue grew 14% [1], mainly in Europe [2] and Asia [1][3], with details in [4, 5]."
    indices = parse_citations(text)
    assert indices == [1, 2, 3, 4, 5]


def test_parse_citations_handles_empty_or_no_citations() -> None:
    assert parse_citations("") == []
    assert parse_citations("No citations here.") == []
    assert parse_citations("Regular brackets [unrelated text]") == []


def test_generate_grounded_answer_empty_chunks_returns_refusal() -> None:
    answer = generate_grounded_answer("What was revenue?", context_chunks=[])
    assert answer.content == NO_INFO_MESSAGE
    assert answer.sources == []
    assert answer.cited_indices == []


def test_generate_grounded_answer_with_mock_client() -> None:
    client = Mock()
    client.generate.return_value = "Revenue increased by 14.2% [1] due to international sales [2]."

    chunks = [
        ChunkSearchResult(id="c1", physical_document_id="p1", chunk_index=0, page_number=12, content="chunk 1", score=0.03, source="doc.pdf"),
        ChunkSearchResult(id="c2", physical_document_id="p1", chunk_index=1, page_number=14, content="chunk 2", score=0.02, source="doc.pdf"),
    ]

    answer = generate_grounded_answer("Revenue?", context_chunks=chunks, client=client)

    client.generate.assert_called_once()
    assert answer.content == "Revenue increased by 14.2% [1] due to international sales [2]."
    assert len(answer.sources) == 2
    assert answer.cited_indices == [1, 2]


def test_generate_grounded_answer_filters_out_of_range_citations() -> None:
    client = Mock()
    # Model cites [1] and non-existent [99]
    client.generate.return_value = "Fact [1] and hallucinated reference [99]."

    chunks = [
        ChunkSearchResult(id="c1", physical_document_id="p1", chunk_index=0, page_number=1, content="chunk 1", score=0.05, source="doc.pdf")
    ]

    answer = generate_grounded_answer("Fact?", context_chunks=chunks, client=client)
    assert answer.cited_indices == [1]


def test_generate_grounded_answer_propagates_client_error() -> None:
    client = Mock()
    client.generate.side_effect = RuntimeError("API quota exceeded")

    chunks = [
        ChunkSearchResult(id="c1", physical_document_id="p1", chunk_index=0, page_number=1, content="chunk 1", score=0.05, source="doc.pdf")
    ]

    with pytest.raises(RuntimeError, match="API quota exceeded"):
        generate_grounded_answer("Question", context_chunks=chunks, client=client)
