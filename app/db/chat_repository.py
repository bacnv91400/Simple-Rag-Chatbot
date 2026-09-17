"""Repository operations for chat sessions, messages, sources, and document tracking."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from supabase import Client, create_client

from app.retrieval.models import ChunkSearchResult

logger = logging.getLogger(__name__)


class ChatRepository:
    """Manages chat persistence and document metadata using Supabase."""

    def __init__(self, client: Client) -> None:
        self.client = client

    def create_session(self, title: str = "New Chat") -> dict[str, Any]:
        """Create a new chat session."""
        response = (
            self.client.table("chat_sessions")
            .insert({"title": title})
            .execute()
        )
        data = getattr(response, "data", None) or []
        if data:
            return data[0]
        raise RuntimeError("Failed to create chat session")

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all chat sessions, most recent first."""
        response = (
            self.client.table("chat_sessions")
            .select("id, title, created_at")
            .order("created_at", desc=True)
            .execute()
        )
        return getattr(response, "data", None) or []

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Fetch a single chat session by ID."""
        response = (
            self.client.table("chat_sessions")
            .select("id, title, created_at")
            .eq("id", session_id)
            .limit(1)
            .execute()
        )
        rows = getattr(response, "data", None) or []
        return rows[0] if rows else None

    def update_session_title(self, session_id: str, title: str) -> None:
        """Update the title of a chat session."""
        self.client.table("chat_sessions").update({"title": title}).eq("id", session_id).execute()

    def delete_session(self, session_id: str) -> None:
        """Delete a chat session (cascades to messages and sources)."""
        self.client.table("chat_sessions").delete().eq("id", session_id).execute()

    def add_message(self, session_id: str, role: str, content: str) -> dict[str, Any]:
        """Insert a single message into the chat session."""
        response = (
            self.client.table("messages")
            .insert({"chat_session_id": session_id, "role": role, "content": content})
            .execute()
        )
        data = getattr(response, "data", None) or []
        if data:
            return data[0]
        raise RuntimeError("Failed to create message")

    def add_message_sources(
        self,
        message_id: str,
        sources: Sequence[ChunkSearchResult | dict[str, Any]],
    ) -> None:
        """Insert source references for an assistant message."""
        if not sources:
            return
        rows: list[dict[str, Any]] = []
        for s in sources:
            chunk_id = getattr(s, "id", None) or (s.get("chunk_id") if isinstance(s, dict) else None)
            score = getattr(s, "score", None) or (s.get("score") if isinstance(s, dict) else None)
            if chunk_id:
                rows.append({
                    "message_id": message_id,
                    "chunk_id": str(chunk_id),
                    "relevance_score": float(score) if score is not None else None,
                })
        if rows:
            self.client.table("message_sources").insert(rows).execute()

    def get_messages_with_sources(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieve all messages for a session with resolved source chunk details."""
        response = (
            self.client.table("messages")
            .select("id, chat_session_id, role, content, created_at")
            .eq("chat_session_id", session_id)
            .order("created_at", desc=False)
            .execute()
        )
        messages: list[dict[str, Any]] = getattr(response, "data", None) or []
        if not messages:
            return []

        message_ids = [m["id"] for m in messages]
        try:
            sources_resp = (
                self.client.table("message_sources")
                .select("id, message_id, chunk_id, relevance_score")
                .in_("message_id", message_ids)
                .execute()
            )
            sources_rows: list[dict[str, Any]] = getattr(sources_resp, "data", None) or []
        except Exception as err:
            logger.warning("Failed to fetch message sources: %s", err)
            sources_rows = []

        chunk_ids = list({r["chunk_id"] for r in sources_rows if r.get("chunk_id")})
        chunk_map: dict[str, dict[str, Any]] = {}
        if chunk_ids:
            try:
                chunks_resp = (
                    self.client.table("document_chunks")
                    .select("id, physical_document_id, page_number, content")
                    .in_("id", chunk_ids)
                    .execute()
                )
                for chunk in getattr(chunks_resp, "data", None) or []:
                    chunk_map[chunk["id"]] = chunk
            except Exception as err:
                logger.warning("Failed to fetch chunk details: %s", err)

        # Resolve filenames for physical documents
        doc_ids = list({c["physical_document_id"] for c in chunk_map.values() if c.get("physical_document_id")})
        doc_name_map: dict[str, str] = {}
        if doc_ids:
            try:
                uploads_resp = (
                    self.client.table("document_uploads")
                    .select("physical_document_id, display_filename")
                    .in_("physical_document_id", doc_ids)
                    .execute()
                )
                for u in getattr(uploads_resp, "data", None) or []:
                    pid = u["physical_document_id"]
                    if pid not in doc_name_map:
                        doc_name_map[pid] = u["display_filename"]
            except Exception as err:
                logger.warning("Failed to resolve document filenames: %s", err)

        # Group sources by message_id
        message_sources_map: dict[str, list[dict[str, Any]]] = {mid: [] for mid in message_ids}
        for src in sources_rows:
            mid = src["message_id"]
            cid = src.get("chunk_id")
            chunk_info = chunk_map.get(cid, {}) if cid else {}
            pid = chunk_info.get("physical_document_id")
            source_display = doc_name_map.get(pid, "Document") if pid else "Document"

            message_sources_map.setdefault(mid, []).append({
                "chunk_id": cid,
                "source": source_display,
                "page_number": chunk_info.get("page_number"),
                "content": chunk_info.get("content", ""),
                "score": src.get("relevance_score"),
            })

        for m in messages:
            m["sources"] = message_sources_map.get(m["id"], [])

        return messages

    def get_all_documents(self) -> list[dict[str, Any]]:
        """Retrieve all uploaded documents with page count and live processing status."""
        try:
            uploads_resp = (
                self.client.table("document_uploads")
                .select("id, physical_document_id, display_filename, created_at")
                .order("created_at", desc=True)
                .execute()
            )
            uploads = getattr(uploads_resp, "data", None) or []
            if not uploads:
                return []

            doc_ids = list({u["physical_document_id"] for u in uploads})
            docs_resp = (
                self.client.table("physical_documents")
                .select("id, page_count, processing_status, processing_error")
                .in_("id", doc_ids)
                .execute()
            )
            doc_map = {d["id"]: d for d in (getattr(docs_resp, "data", None) or [])}

            results: list[dict[str, Any]] = []
            for u in uploads:
                pid = u["physical_document_id"]
                pdoc = doc_map.get(pid, {})
                results.append({
                    "id": u["id"],
                    "physical_document_id": pid,
                    "display_filename": u["display_filename"],
                    "created_at": u.get("created_at"),
                    "page_count": pdoc.get("page_count", 0),
                    "processing_status": pdoc.get("processing_status", "pending"),
                    "processing_error": pdoc.get("processing_error"),
                })
            return results
        except Exception as err:
            logger.warning("Failed to fetch all documents: %s", err)
            return []


def create_chat_repository(url: str, service_role_key: str) -> ChatRepository:
    """Factory helper to instantiate ChatRepository."""
    return ChatRepository(create_client(url, service_role_key))
