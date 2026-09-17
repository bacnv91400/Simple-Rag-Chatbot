"""Chat conversation interface with grounded answer generation and multi-turn persistence."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import streamlit as st
from supabase import create_client

from app.config.settings import settings
from app.db.chat_repository import ChatRepository
from app.generation.answer import GroundedAnswer, generate_grounded_answer
from app.generation.client import GeminiChatClient
from app.ingestion.embedder import GeminiEmbedder, MinIntervalRateLimiter
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.models import ChunkSearchResult
from app.ui.sources import render_sources

logger = logging.getLogger(__name__)

SUGGESTED_QUESTIONS = [
    "Summarize the key findings across the documents",
    "What are the main numbers and metrics reported?",
    "What are the primary challenges or risks mentioned?",
]


def _populate_sources(
    results: list[ChunkSearchResult],
    client,
) -> list[ChunkSearchResult]:
    """Join chunk results to document_uploads to populate display filenames."""
    if not results:
        return results

    doc_ids = list({r.physical_document_id for r in results})
    try:
        response = (
            client.table("document_uploads")
            .select("physical_document_id,display_filename")
            .in_("physical_document_id", doc_ids)
            .execute()
        )
        rows = getattr(response, "data", None) or []
        name_map: dict[str, str] = {row["physical_document_id"]: row["display_filename"] for row in rows}

        for result in results:
            result.source = name_map.get(result.physical_document_id, "Unknown document")
    except Exception as error:
        logger.warning("Failed to resolve source filenames: %s", error)
        for result in results:
            result.source = "Unknown document"
    return results


def run_hybrid_retrieval(question: str) -> list[ChunkSearchResult]:
    """Embed the question and execute hybrid vector + BM25 search with RRF fusion."""
    embedder = GeminiEmbedder(
        api_key=settings.google_api_key or "",
        model=settings.gemini_embed_model,
        output_dim=settings.embed_output_dim,
        rate_limiter=MinIntervalRateLimiter(settings.embed_min_seconds_between_requests),
        max_retries=settings.embed_max_retries,
        retry_base_delay=settings.embed_retry_base_delay,
    )
    query_embedding = embedder.embed_one(question, "RETRIEVAL_QUERY")

    client = create_client(
        settings.supabase_url or "",
        settings.supabase_service_role_key or "",
    )

    results = hybrid_search(
        client,
        query=question,
        query_embedding=query_embedding,
        top_k=settings.retrieval_top_k,
        rrf_k=settings.retrieval_rrf_k,
        overfetch_multiplier=settings.retrieval_overfetch_multiplier,
        min_similarity=settings.retrieval_similarity_threshold,
    )
    return _populate_sources(results, client)


def _generate_title(question: str) -> str:
    """Generate a clean, readable session title from the first question."""
    cleaned = question.strip()
    if len(cleaned) <= 35:
        return cleaned
    # Cut off cleanly at the nearest word boundary
    truncated = cleaned[:35].rsplit(" ", 1)[0]
    return f"{truncated}..."


def render_chat(chat_repo: ChatRepository) -> None:
    """Render the main conversation interface for the active session."""
    active_session_id = st.session_state.get("active_session_id")
    if not active_session_id:
        return

    messages = chat_repo.get_messages_with_sources(active_session_id)

    # Empty state rendering
    if not messages:
        st.markdown("## Simple RAG Chatbot")
        st.markdown(
            "Ask questions grounded in your uploaded documents. "
            "Answers cite supporting document chunks with transparent references."
        )
        st.write("")

        st.caption("Suggested questions to get started:")
        cols = st.columns(len(SUGGESTED_QUESTIONS))
        for idx, suggestion in enumerate(SUGGESTED_QUESTIONS):
            if cols[idx].button(suggestion, key=f"suggestion_{idx}", use_container_width=True):
                st.session_state["pending_prompt"] = suggestion
                st.rerun()

    # Render previous conversation history
    for msg in messages:
        role = msg["role"]
        with st.chat_message(role):
            st.markdown(msg["content"])
            if role == "assistant" and msg.get("sources"):
                render_sources(msg["sources"])

    # Determine input prompt (either from chat_input or clicked suggestion)
    user_input = st.chat_input("Ask a question about your documents...")
    prompt: str | None = None
    if user_input:
        prompt = user_input
    elif "pending_prompt" in st.session_state:
        prompt = st.session_state.pop("pending_prompt")

    if prompt and prompt.strip():
        prompt = prompt.strip()

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # Save user message
        chat_repo.add_message(active_session_id, "user", prompt)

        # If first message, update session title
        if not messages:
            new_title = _generate_title(prompt)
            try:
                chat_repo.update_session_title(active_session_id, new_title)
            except Exception as err:
                logger.warning("Failed to update session title: %s", err)

        # Process answer generation in assistant container
        with st.chat_message("assistant"):
            # Step 1: Hybrid Retrieval
            retrieval_failed = False
            retrieved_chunks: list[ChunkSearchResult] = []
            with st.spinner("Searching your documents..."):
                try:
                    retrieved_chunks = run_hybrid_retrieval(prompt)
                except Exception as error:
                    logger.error("Retrieval error: %s", error)
                    retrieval_failed = True
                    st.error(f"Failed to search documents: {error}")

            if retrieval_failed:
                return

            # Step 2: Answer Generation
            answer: GroundedAnswer
            with st.spinner("Generating answer..."):
                try:
                    gemini_client = GeminiChatClient(
                        api_key=settings.google_api_key or "",
                        model=settings.gemini_chat_model,
                    )
                    # Prepare history for multi-turn context
                    history_payload = [
                        {"role": m["role"], "content": m["content"]}
                        for m in messages
                    ]
                    answer = generate_grounded_answer(
                        question=prompt,
                        context_chunks=retrieved_chunks,
                        history=history_payload,
                        client=gemini_client,
                    )
                except Exception as error:
                    logger.error("Generation error: %s", error)
                    st.error(f"Failed to generate answer: {error}")
                    return

            st.markdown(answer.content)
            if answer.sources:
                render_sources(answer.sources)

            # Persist assistant message and sources
            try:
                saved_msg = chat_repo.add_message(active_session_id, "assistant", answer.content)
                if answer.sources:
                    chat_repo.add_message_sources(saved_msg["id"], answer.sources)
            except Exception as error:
                logger.warning("Failed to save assistant message: %s", error)

        # Rerun to cleanly synchronize state
        st.rerun()
