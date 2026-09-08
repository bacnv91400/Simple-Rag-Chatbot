"""Live processing-state component for uploaded physical documents."""

from __future__ import annotations

from typing import Any

import streamlit as st

from app.config.settings import settings
from app.db.chunks_repository import create_chunks_repository

_STATUS_LABELS = {
    "pending": ("Queued", "⏳"),
    "processing": ("Processing", "⚙️"),
    "completed": ("Ready", "✅"),
    "failed": ("Failed", "❌"),
}


@st.fragment(run_every=3)
def render_document_statuses(uploads: list[dict[str, Any]]) -> None:
    """Poll status without rerunning the upload controls or the rest of the UI."""
    st.subheader("Processing status")
    document_ids = [upload["physical_document_id"] for upload in uploads]
    if not document_ids:
        st.caption("Upload a PDF to queue background processing.")
        return
    try:
        repository = create_chunks_repository(
            settings.supabase_url or "", settings.supabase_service_role_key or ""
        )
        statuses = repository.get_statuses(document_ids)
    except Exception as error:
        st.warning(f"Unable to refresh processing status: {error}")
        return

    for upload in uploads:
        document_id = upload["physical_document_id"]
        row = statuses.get(document_id, {"processing_status": "pending"})
        status = row["processing_status"]
        label, icon = _STATUS_LABELS.get(status, (status.title(), "•"))
        left, right = st.columns([5, 1])
        left.write(f"{icon} **{upload['display_filename']}** — {label}")
        if status == "failed":
            error = row.get("processing_error") or "No error details were recorded."
            left.caption(error)
            if right.button("Retry", key=f"retry:{document_id}"):
                repository.retry(document_id)
                st.rerun(scope="fragment")
