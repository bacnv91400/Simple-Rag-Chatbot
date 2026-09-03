"""Streamlit components for the user flow UI."""

from collections.abc import Sequence

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from app.config.settings import settings


def render_uploaded_documents(uploaded_files: Sequence[UploadedFile]) -> None:
    """Show the selected PDF filenames."""
    st.subheader("Uploaded documents")
    if not uploaded_files:
        st.caption("No documents selected yet.")
        return

    for uploaded_file in uploaded_files:
        st.write(f"- {uploaded_file.name}")


def render_status(has_documents: bool) -> None:
    """Display the processing state."""
    st.subheader("Status")
    if has_documents:
        st.info("Ready to process.")
    else:
        st.info("Upload one or more PDF documents to prepare for processing.")


def render_question_form() -> tuple[bool, str]:
    """Render the question field and return its submit state and value."""
    st.subheader("Ask a question")
    with st.form("question_form"):
        question = st.text_input(
            "Question",
            placeholder="What is this document about?",
        )
        submitted = st.form_submit_button("Ask")
    return submitted, question


def render_answer(submitted: bool, question: str) -> None:
    """Render a deliberately non-functional answer area."""
    st.subheader("Answer")
    if submitted and question.strip():
        st.info("RAG answer...")
    elif submitted:
        st.warning("Enter a question before selecting Ask.")
    else:
        st.caption("please enter a question and select Ask to see an answer.")


def render_sources() -> None:
    """Render the placeholder for retrieval citations."""
    st.subheader("Sources")
    st.caption("Source references!")


def render() -> None:
    """Render the Simple RAG Chatbot interface."""
    st.set_page_config(page_title=settings.app_name, layout="centered")
    st.title(settings.app_name)
    st.caption("Upload PDF documents and ask questions grounded in their content.")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True,
    )
    render_uploaded_documents(uploaded_files)
    render_status(bool(uploaded_files))

    st.divider()
    submitted, question = render_question_form()
    render_answer(submitted, question)
    render_sources()
