"""Data models for retrieval search results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChunkSearchResult:
    """A single chunk returned from vector, BM25, or hybrid search."""

    id: str
    physical_document_id: str
    chunk_index: int
    page_number: int | None
    content: str
    score: float
    source: str = field(default="")
