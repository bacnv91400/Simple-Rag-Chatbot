"""Repository operations owned by the background ingestion worker."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from supabase import Client, create_client

logger = logging.getLogger(__name__)


class ImageCaptionCache:
    """Postgres-backed cache for Gemini image captions."""

    def __init__(self, client: Client) -> None:
        self.client = client

    def get(self, image_hash: str) -> str | None:
        try:
            response = (
                self.client.table("image_caption_cache")
                .select("caption")
                .eq("image_hash", image_hash)
                .limit(1)
                .execute()
            )
            data = getattr(response, "data", None) or []
            if data:
                return data[0].get("caption")
        except Exception as error:
            logger.warning("Failed to read image caption cache for %s: %s", image_hash, error)
        return None

    def set(self, image_hash: str, caption: str) -> None:
        try:
            self.client.table("image_caption_cache").upsert(
                {"image_hash": image_hash, "caption": caption},
                on_conflict="image_hash",
            ).execute()
        except Exception as error:
            logger.warning("Failed to write image caption cache for %s: %s", image_hash, error)


class ChunksRepository:
    def __init__(self, client: Client) -> None:
        self.client = client
        self.image_cache = ImageCaptionCache(client)

    def claim_pending(self, batch_size: int) -> list[dict[str, Any]]:
        response = self.client.rpc(
            "claim_pending_documents", {"p_batch_size": batch_size}
        ).execute()
        return getattr(response, "data", None) or []

    def save_chunks(self, physical_document_id: str, records: list[Any]) -> None:
        if not records:
            raise ValueError("Document produced no non-empty chunks")
        rows = [
            {
                "physical_document_id": physical_document_id,
                "chunk_index": getattr(record, "index", getattr(record, "chunk_index", 0)),
                "page_number": getattr(record, "page_start", getattr(record, "page_number", None)),
                "page_end": getattr(record, "page_end", getattr(record, "page_start", getattr(record, "page_number", None))),
                "section": getattr(record, "section", ""),
                "content": getattr(record, "text", getattr(record, "content", "")),
                "content_type": getattr(record, "content_type", "text"),
                "token_count": getattr(record, "token_count", None),
                "overlap_from_chunk_index": getattr(record, "overlap_from_chunk_index", None),
                "embedding": record.embedding,
            }
            for record in records
        ]
        # PostgREST translates this to INSERT ... ON CONFLICT ... DO UPDATE.
        self.client.table("document_chunks").upsert(
            rows, on_conflict="physical_document_id,chunk_index"
        ).execute()

    def mark_completed(self, physical_document_id: str) -> None:
        self.client.table("physical_documents").update({
            "processing_status": "completed", "processing_error": None,
            "updated_at": datetime.now(UTC).isoformat(),
        }).eq("id", physical_document_id).execute()

    def mark_failed(self, physical_document_id: str, error: str) -> None:
        self.client.table("physical_documents").update({
            "processing_status": "failed", "processing_error": error[:4_000],
            "updated_at": datetime.now(UTC).isoformat(),
        }).eq("id", physical_document_id).execute()

    def sweep_stale_processing(self, minutes: int) -> None:
        cutoff = datetime.now(UTC) - timedelta(minutes=minutes)
        self.client.table("physical_documents").update({
            "processing_status": "pending", "updated_at": datetime.now(UTC).isoformat(),
        }).eq("processing_status", "processing").lt("updated_at", cutoff.isoformat()).execute()

    def get_statuses(self, physical_document_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not physical_document_ids:
            return {}
        response = self.client.table("physical_documents").select(
            "id,processing_status,processing_error"
        ).in_("id", physical_document_ids).execute()
        return {row["id"]: row for row in (getattr(response, "data", None) or [])}

    def retry(self, physical_document_id: str) -> None:
        self.client.table("physical_documents").update({
            "processing_status": "pending", "processing_error": None,
            "updated_at": datetime.now(UTC).isoformat(),
        }).eq("id", physical_document_id).eq("processing_status", "failed").execute()


def create_chunks_repository(url: str, service_role_key: str) -> ChunksRepository:
    return ChunksRepository(create_client(url, service_role_key))
