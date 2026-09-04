"""Minimal server-side Supabase access for the ingestion pipeline."""

from __future__ import annotations

from typing import Any

from supabase import Client, create_client


class DocumentRepository:
    def __init__(self, client: Client) -> None:
        self.client = client

    def find_physical_document(self, file_hash: str) -> dict[str, Any] | None:
        # Use a list response: Supabase/PostgREST represents no match as an
        # empty list, avoiding client-version differences around maybe_single.
        response = (
            self.client.table("physical_documents")
            .select("*")
            .eq("file_hash", file_hash)
            .limit(1)
            .execute()
        )
        rows = getattr(response, "data", None) or []
        return rows[0] if rows else None

    def save_successful_upload(
        self, *, file_hash: str, storage_key: str, file_size_bytes: int, page_count: int,
        requested_filename: str, is_duplicate_content: bool,
    ) -> dict[str, Any]:
        response = self.client.rpc("save_successful_upload", {
            "p_file_hash": file_hash, "p_storage_key": storage_key,
            "p_file_size_bytes": file_size_bytes, "p_page_count": page_count,
            "p_requested_filename": requested_filename,
            "p_is_duplicate_content": is_duplicate_content,
        }).execute()
        return response.data[0] if isinstance(response.data, list) else response.data


def create_repository(url: str, service_role_key: str) -> DocumentRepository:
    return DocumentRepository(create_client(url, service_role_key))
