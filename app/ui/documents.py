"""Document management and live status monitoring component."""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

from app.config.settings import settings
from app.db.chat_repository import ChatRepository
from app.db.chunks_repository import ChunksRepository, create_chunks_repository
from app.ingestion.pipeline import ProcessUploadResult, process_upload

logger = logging.getLogger(__name__)

_STATUS_CONFIG = {
    "pending": ("Queued", "⏳", "badge-pending"),
    "processing": ("Processing", "⚙️", "badge-processing"),
    "completed": ("Ready", "✅", "badge-ready"),
    "failed": ("Failed", "❌", "badge-failed"),
}


def handle_document_uploads() -> list[dict[str, Any]]:
    """Render the file uploader and process any newly uploaded PDFs."""
    with st.expander("📤 Upload PDF Documents", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type="pdf",
            accept_multiple_files=True,
            key="pdf_uploader",
        )

        missing = settings.missing_upload_settings()
        if missing:
            st.error(f"Upload configuration is incomplete: {', '.join(missing)}")
            return []

        if not uploaded_files:
            return []

        recent_uploads: list[dict[str, Any]] = []
        for uploaded_file in uploaded_files:
            state_key = f"upload-result:{uploaded_file.file_id}"
            if state_key not in st.session_state:
                st.session_state[state_key] = process_upload(
                    uploaded_file.name, uploaded_file.getvalue(), settings=settings
                )
            result: ProcessUploadResult = st.session_state[state_key]

            if result.success:
                st.success(f"Uploaded: {result.upload['display_filename']}")
                recent_uploads.append(result.upload)
            else:
                st.error(f"Failed: {uploaded_file.name} - {result.error or 'Error'}")

            # Expandable process logs
            with st.expander(f"Logs: {uploaded_file.name}", expanded=not result.success):
                for entry in result.logs:
                    icon = {"success": "✅", "failed": "❌", "skipped": "⏭️"}.get(entry.status, "•")
                    st.write(f"{icon} `{entry.step}` — {entry.message}")

        return recent_uploads


@st.fragment(run_every=3)
def render_document_list(chat_repo: ChatRepository) -> None:
    """Render the list of uploaded documents and their statuses, polling live."""
    st.markdown("### Documents")

    chunks_repo: ChunksRepository | None = None
    if settings.supabase_url and settings.supabase_service_role_key:
        chunks_repo = create_chunks_repository(
            settings.supabase_url, settings.supabase_service_role_key
        )

    all_docs = chat_repo.get_all_documents()
    if not all_docs:
        st.caption("No documents uploaded yet.")
        return

    # Check live status from physical_documents
    doc_ids = [d["physical_document_id"] for d in all_docs if d.get("physical_document_id")]
    live_statuses: dict[str, dict[str, Any]] = {}
    if chunks_repo and doc_ids:
        try:
            live_statuses = chunks_repo.get_statuses(doc_ids)
        except Exception as err:
            logger.warning("Failed to refresh document statuses: %s", err)

    for doc in all_docs:
        pid = doc.get("physical_document_id")
        live_info = live_statuses.get(pid, {}) if pid else {}
        status_key = live_info.get("processing_status") or doc.get("processing_status", "pending")
        label, icon, _ = _STATUS_CONFIG.get(status_key, (status_key.title(), "•", ""))
        pages = doc.get("page_count", 0)

        filename = doc.get("display_filename", "Document")
        meta_line = f"{pages} pages · {label}" if pages else label

        with st.container():
            st.markdown(
                f"""
                <div class="doc-card">
                    <div class="doc-card-title">{icon} {filename}</div>
                    <div class="doc-card-meta">{meta_line}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if status_key == "failed":
                error_msg = live_info.get("processing_error") or doc.get("processing_error") or "Unknown error"
                st.caption(f"Error: {error_msg}")
                if chunks_repo and pid and st.button("Retry", key=f"retry_{pid}"):
                    chunks_repo.retry(pid)
                    st.rerun(scope="fragment")
