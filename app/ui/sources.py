"""Component for rendering expandable grounded source citations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import streamlit as st

from app.retrieval.models import ChunkSearchResult


def render_sources(sources: Sequence[ChunkSearchResult | dict[str, Any]]) -> None:
    """Render expandable source cards for an assistant response."""
    if not sources:
        return

    with st.expander("📚 Sources", expanded=False):
        for idx, src in enumerate(sources, start=1):
            if isinstance(src, ChunkSearchResult):
                source_name = src.source or "Document"
                page_num = src.page_number
                content = src.content
                score = src.score
            else:
                source_name = src.get("source") or "Document"
                page_num = src.get("page_number")
                content = src.get("content", "")
                score = src.get("score")

            page_label = f" · Page {page_num}" if page_num is not None else ""
            score_label = f" · RRF score: {score:.4f}" if score is not None else ""
            header = f"**[{idx}] {source_name}**{page_label}{score_label}"

            st.markdown(header)
            # Show preview or expandable full text
            st.text_area(
                label=f"Source [{idx}] content",
                value=content.strip(),
                height=120,
                disabled=True,
                key=f"source_view_{id(src)}_{idx}",
                label_visibility="collapsed",
            )
            if idx < len(sources):
                st.divider()
