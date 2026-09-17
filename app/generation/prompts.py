"""Prompt engineering and formatting for grounded RAG generation."""

from __future__ import annotations

from collections.abc import Sequence

from app.retrieval.models import ChunkSearchResult

SYSTEM_INSTRUCTION = """You are a knowledgeable, precise document assistant for the Simple RAG Chatbot.
Your goal is to answer the user's question accurately using ONLY the provided document context chunks.

STRICT GROUNDING & CITATION RULES:
1. Grounding: Rely EXCLUSIVELY on facts directly mentioned in the provided Context Chunks. Do NOT extrapolate, speculate, or introduce external knowledge.
2. Insufficient Context: If the provided chunks do not contain enough information to answer the question, state exactly:
   "I couldn't find enough information in your documents to answer this question."
   Do NOT attempt to guess or partially hallucinate an answer.
3. Citations: Every factual assertion, number, date, or claim MUST cite the supporting chunk(s) using square brackets, e.g., [1], [2], or [1][2].
   - Use the index corresponding to the numbered context chunk (e.g. [1] for chunk 1).
   - Place citation marks directly after the relevant statement or sentence.
4. Synthesis: Synthesize information smoothly across multiple chunks into a clear, coherent response. Preserve exact numbers, dates, proper names, and technical terms.
5. Conversation Flow: Use the conversation history to understand context and follow-up references (e.g. pronouns or relative comparisons), but ALWAYS ground any answers in the provided Context Chunks.
"""


def format_context_chunks(chunks: Sequence[ChunkSearchResult]) -> str:
    """Format retrieved chunks into numbered context blocks for prompt injection."""
    if not chunks:
        return "No document context available."

    formatted_blocks: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        source_name = chunk.source or "Document"
        page_info = f" · Page {chunk.page_number}" if chunk.page_number is not None else ""
        header = f"[{idx}] {source_name}{page_info}"
        content = chunk.content.strip()
        formatted_blocks.append(f"{header}\n{content}")

    return "\n\n".join(formatted_blocks)


def format_chat_history(history: Sequence[dict[str, str]], max_turns: int = 6) -> str:
    """Format recent conversation history turns."""
    if not history:
        return ""

    recent = history[-max_turns:]
    lines: list[str] = []
    for msg in recent:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "").strip()
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def build_user_prompt(
    question: str,
    context_chunks: Sequence[ChunkSearchResult],
    history: Sequence[dict[str, str]] | None = None,
) -> str:
    """Assemble the complete prompt containing context chunks, history, and current question."""
    context_text = format_context_chunks(context_chunks)
    history_text = format_chat_history(history or [])

    parts: list[str] = [
        "--- DOCUMENT CONTEXT CHUNKS ---",
        context_text,
        "--- END OF CONTEXT ---",
    ]

    if history_text:
        parts.extend([
            "",
            "--- CONVERSATION HISTORY ---",
            history_text,
            "--- END OF HISTORY ---",
        ])

    parts.extend([
        "",
        f"User Question: {question}",
        "",
        "Instructions: Provide a clear, grounded answer citing supporting sources with [1], [2], etc.",
    ])

    return "\n".join(parts)
