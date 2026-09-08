"""Single-document background processing orchestration."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import boto3

from app.config.settings import Settings
from app.db.chunks_repository import ChunksRepository
from app.ingestion.chunker import chunk_document
from app.ingestion.embedder import GeminiEmbedder, MinIntervalRateLimiter
from app.ingestion.extractor import extract_to_docling_document

logger = logging.getLogger(__name__)


def _s3_client(settings: Settings) -> Any:
    return boto3.client(
        "s3",
        endpoint_url=settings.idrive_e2_endpoint,
        region_name=settings.idrive_e2_region,
        aws_access_key_id=settings.idrive_e2_access_key,
        aws_secret_access_key=settings.idrive_e2_secret_key,
    )


def process_document(
    physical_document_id: str,
    storage_key: str,
    *,
    settings: Settings,
    repository: ChunksRepository,
    embedder: GeminiEmbedder | None = None,
    rate_limiter: MinIntervalRateLimiter | None = None,
    s3_client: Any | None = None,
    converter: Any | None = None,
) -> None:
    """Download, extract, chunk, embed, persist, and finalise one claimed PDF."""
    try:
        limiter = rate_limiter or MinIntervalRateLimiter(settings.embed_min_seconds_between_requests)
        active_embedder = embedder or GeminiEmbedder(
            api_key=settings.google_api_key or "",
            model=settings.gemini_embed_model,
            output_dim=settings.embed_output_dim,
            rate_limiter=limiter,
            max_retries=settings.embed_max_retries,
            retry_base_delay=settings.embed_retry_base_delay,
        )

        with tempfile.TemporaryDirectory(prefix="pdf-rag-") as directory:
            file_path = Path(directory) / "source.pdf"
            client = s3_client or _s3_client(settings)
            client.download_file(settings.idrive_e2_bucket, storage_key, str(file_path))

            # If converter is provided (e.g. in test mock), use its convert method
            if converter is not None and hasattr(converter, "convert"):
                document = converter.convert(file_path).document
            else:
                document = extract_to_docling_document(
                    pdf_path=file_path,
                    gemini_client=getattr(active_embedder, "client", None),
                    cache=repository.image_cache,
                    rate_limiter=limiter,
                    max_gemini_calls=settings.max_gemini_calls_per_doc,
                    caption_model=settings.gemini_caption_model,
                    min_image_width=settings.min_image_width,
                    min_image_height=settings.min_image_height,
                    min_image_area_ratio=settings.min_image_area_ratio,
                )

            chunks = chunk_document(
                doc=document,
                tokenizer_model=settings.chunk_tokenizer_model or "sentence-transformers/all-MiniLM-L6-v2",
                max_tokens=settings.chunk_max_tokens,
                overlap_tokens=settings.chunk_overlap_tokens,
            )

            for chunk in chunks:
                chunk.embedding = active_embedder.embed_one(chunk.text, task_type="RETRIEVAL_DOCUMENT")

            repository.save_chunks(physical_document_id, chunks)

        repository.mark_completed(physical_document_id)
        logger.info("Successfully processed and indexed document %s", physical_document_id)
    except Exception as error:
        logger.exception("Failed to process document %s: %s", physical_document_id, error)
        try:
            repository.mark_failed(physical_document_id, str(error))
        except Exception as mark_error:
            raise RuntimeError(
                f"Processing failed ({error}); also unable to persist failure ({mark_error})"
            ) from mark_error
