"""Grounded answer generation orchestrator and citation parsing."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.generation.client import GeminiChatClient
from app.generation.prompts import build_user_prompt
from app.retrieval.models import ChunkSearchResult

logger = logging.getLogger(__name__)

NO_INFO_MESSAGE = "I couldn't find enough information in your documents to answer this question."


@dataclass
class GroundedAnswer:
    """Encapsulates the generated answer and associated citation sources."""

    content: str
    sources: list[ChunkSearchResult] = field(default_factory=list)
    cited_indices: list[int] = field(default_factory=list)


def parse_citations(text: str) -> list[int]:
    """Extract unique 1-based citation indices [1], [2], [1, 2], [1][2] from answer text."""
    if not text:
        return []

    # Matches bracketed numbers like [1], [2], [1, 2], [1, 3, 5]
    matches = re.findall(r"\[([0-9,\s]+)\]", text)
    indices: set[int] = set()
    for match in matches:
        for part in match.split(","):
            cleaned = part.strip()
            if cleaned.isdigit():
                indices.add(int(cleaned))
    return sorted(indices)


def generate_grounded_answer(
    question: str,
    context_chunks: Sequence[ChunkSearchResult],
    history: Sequence[dict[str, str]] | None = None,
    client: GeminiChatClient | None = None,
) -> GroundedAnswer:
    """Generate a strictly grounded answer from retrieved context chunks using Gemini."""
    if not context_chunks:
        return GroundedAnswer(
            content=NO_INFO_MESSAGE,
            sources=[],
            cited_indices=[],
        )

    if client is None:
        raise ValueError("GeminiChatClient must be provided for answer generation")

    prompt = build_user_prompt(question, context_chunks, history)
    response_text = client.generate(prompt)

    cited_indices = parse_citations(response_text)
    # Restrict cited indices to those in valid range of chunks
    valid_indices = [idx for idx in cited_indices if 1 <= idx <= len(context_chunks)]

    return GroundedAnswer(
        content=response_text.strip(),
        sources=list(context_chunks),
        cited_indices=valid_indices,
    )
