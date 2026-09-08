"""Tests for worker processing orchestration."""

from unittest.mock import Mock

from app.config.settings import Settings
from app.ingestion.chunker import Chunk
from worker import processing


def _settings() -> Settings:
    return Settings(
        idrive_e2_bucket="bucket",
        chunk_tokenizer_model="tokenizer",
        google_api_key="test-key",
        gemini_embed_model="gemini-embedding-001",
        gemini_caption_model="gemini-2.0-flash-lite",
        embedding_dim=1024,
        embed_output_dim=1024,
    )


def test_process_document_marks_completed_after_saving_chunks(monkeypatch) -> None:
    s3_client = Mock()
    repository = Mock()
    repository.image_cache = Mock()
    mock_embedder = Mock()
    mock_embedder.embed_one.return_value = [0.1] * 1024

    test_chunk = Chunk(
        index=0,
        text="Test content",
        page_start=1,
        page_end=1,
        section="Intro",
        content_type="text",
        token_count=10,
    )

    monkeypatch.setattr(processing, "extract_to_docling_document", Mock(return_value=object()))
    monkeypatch.setattr(processing, "chunk_document", Mock(return_value=[test_chunk]))

    processing.process_document(
        "document-id",
        "document.pdf",
        settings=_settings(),
        repository=repository,
        embedder=mock_embedder,
        s3_client=s3_client,
    )

    mock_embedder.embed_one.assert_called_once_with("Test content", task_type="RETRIEVAL_DOCUMENT")
    repository.save_chunks.assert_called_once()
    saved_chunks = repository.save_chunks.call_args[0][1]
    assert len(saved_chunks) == 1
    assert saved_chunks[0].embedding == [0.1] * 1024

    repository.mark_completed.assert_called_once_with("document-id")
    repository.mark_failed.assert_not_called()


def test_process_document_marks_failed_when_extraction_fails(monkeypatch) -> None:
    s3_client = Mock()
    repository = Mock()
    repository.image_cache = Mock()
    mock_embedder = Mock()

    monkeypatch.setattr(
        processing, "extract_to_docling_document", Mock(side_effect=ValueError("corrupt PDF"))
    )

    processing.process_document(
        "document-id",
        "document.pdf",
        settings=_settings(),
        repository=repository,
        embedder=mock_embedder,
        s3_client=s3_client,
    )

    repository.mark_failed.assert_called_once_with("document-id", "corrupt PDF")
    repository.mark_completed.assert_not_called()
