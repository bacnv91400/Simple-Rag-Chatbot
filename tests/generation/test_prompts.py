"""Tests for prompt engineering and context formatting."""

from app.generation.prompts import (
    SYSTEM_INSTRUCTION,
    build_user_prompt,
    format_chat_history,
    format_context_chunks,
)
from app.retrieval.models import ChunkSearchResult


def test_format_context_chunks_empty() -> None:
    assert format_context_chunks([]) == "No document context available."


def test_format_context_chunks_numbered() -> None:
    chunks = [
        ChunkSearchResult(
            id="c1",
            physical_document_id="p1",
            chunk_index=0,
            page_number=12,
            content="Revenue increased by 14.2%.",
            score=0.03,
            source="Annual Report.pdf",
        ),
        ChunkSearchResult(
            id="c2",
            physical_document_id="p1",
            chunk_index=1,
            page_number=None,
            content="International sales led growth.",
            score=0.02,
            source="",
        ),
    ]
    formatted = format_context_chunks(chunks)
    assert "[1] Annual Report.pdf · Page 12" in formatted
    assert "Revenue increased by 14.2%." in formatted
    assert "[2] Document" in formatted
    assert "International sales led growth." in formatted


def test_format_chat_history() -> None:
    history = [
        {"role": "user", "content": "What was the revenue?"},
        {"role": "assistant", "content": "Revenue was $12.4M [1]."},
    ]
    formatted = format_chat_history(history)
    assert "User: What was the revenue?" in formatted
    assert "Assistant: Revenue was $12.4M [1]." in formatted


def test_build_user_prompt_contains_all_components() -> None:
    chunks = [
        ChunkSearchResult(
            id="c1",
            physical_document_id="p1",
            chunk_index=0,
            page_number=1,
            content="Test content",
            score=0.05,
            source="doc.pdf",
        )
    ]
    history = [{"role": "user", "content": "Prev question"}]
    prompt = build_user_prompt("Current question", chunks, history)

    assert "--- DOCUMENT CONTEXT CHUNKS ---" in prompt
    assert "[1] doc.pdf · Page 1" in prompt
    assert "Test content" in prompt
    assert "--- CONVERSATION HISTORY ---" in prompt
    assert "User: Prev question" in prompt
    assert "User Question: Current question" in prompt


def test_system_instruction_grounding_rules() -> None:
    assert "STRICT GROUNDING & CITATION RULES" in SYSTEM_INSTRUCTION
    assert "I couldn't find enough information in your documents to answer this question." in SYSTEM_INSTRUCTION
    assert "[1]" in SYSTEM_INSTRUCTION
