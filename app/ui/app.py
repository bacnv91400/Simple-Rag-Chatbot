"""Main Streamlit application layout and state coordination."""

from __future__ import annotations

import logging
import streamlit as st

from app.config.settings import settings
from app.db.chat_repository import create_chat_repository
from app.ui.chat import render_chat
from app.ui.documents import handle_document_uploads, render_document_list
from app.ui.styles import apply_custom_styles

logger = logging.getLogger(__name__)


def init_session_state(chat_repo) -> None:
    """Initialize app-level session state variables."""
    if "active_session_id" not in st.session_state:
        sessions = chat_repo.list_sessions()
        if sessions:
            st.session_state["active_session_id"] = sessions[0]["id"]
        else:
            new_session = chat_repo.create_session("New Chat")
            st.session_state["active_session_id"] = new_session["id"]


def render_sidebar(chat_repo) -> None:
    """Render the sidebar containing New Chat, Chat History, and Documents."""
    with st.sidebar:
        st.markdown("## Simple RAG Chatbot")

        # New Chat Button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            new_session = chat_repo.create_session("New Chat")
            st.session_state["active_session_id"] = new_session["id"]
            st.rerun()

        st.divider()

        # Chat History List
        st.markdown("### Chat History")
        sessions = chat_repo.list_sessions()
        active_id = st.session_state.get("active_session_id")

        if not sessions:
            st.caption("No chat history yet.")
        else:
            for s in sessions:
                sid = s["id"]
                title = s.get("title") or "New Chat"
                is_active = sid == active_id

                col1, col2 = st.columns([5, 1])
                button_type = "secondary" if not is_active else "primary"
                prefix = "💬 " if not is_active else "👉 "

                if col1.button(
                    f"{prefix}{title}",
                    key=f"session_btn_{sid}",
                    use_container_width=True,
                    type=button_type,
                ):
                    st.session_state["active_session_id"] = sid
                    st.rerun()

                if col2.button("🗑️", key=f"del_session_{sid}", help="Delete chat"):
                    chat_repo.delete_session(sid)
                    if sid == active_id:
                        remaining = [sess for sess in sessions if sess["id"] != sid]
                        st.session_state["active_session_id"] = remaining[0]["id"] if remaining else None
                    st.rerun()

        st.divider()

        # Document Uploads & Live Status
        handle_document_uploads()
        render_document_list(chat_repo)


def render() -> None:
    """Streamlit entry point for Simple RAG Chatbot."""
    st.set_page_config(
        page_title=settings.app_name,
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_custom_styles()

    if not settings.supabase_url or not settings.supabase_service_role_key:
        st.error("Supabase configuration is incomplete. Please check SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.")
        return

    chat_repo = create_chat_repository(
        settings.supabase_url,
        settings.supabase_service_role_key,
    )

    init_session_state(chat_repo)
    render_sidebar(chat_repo)
    render_chat(chat_repo)
