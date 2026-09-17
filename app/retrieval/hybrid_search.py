"""Hybrid search combining vector similarity and BM25 via Reciprocal Rank Fusion."""

from __future__ import annotations

from supabase import Client

from app.retrieval.bm25_search import bm25_search
from app.retrieval.models import ChunkSearchResult
from app.retrieval.vector_search import similarity_search


def _merge_ranked_results(
    vector_results: list[ChunkSearchResult],
    bm25_results: list[ChunkSearchResult],
    rrf_k: int = 60,
) -> list[ChunkSearchResult]:
    """Merge two ranked result lists using Reciprocal Rank Fusion (RRF).

    Each result receives a score of ``1 / (rrf_k + rank)`` from each list it
    appears in.  Scores are summed across lists and results are returned in
    descending fused-score order.  The ``rrf_k`` constant
    controls how much weight is given to lower-ranked items.
    """
    scores: dict[str, float] = {}
    best: dict[str, ChunkSearchResult] = {}

    for rank, result in enumerate(vector_results, start=1):
        scores[result.id] = scores.get(result.id, 0.0) + 1.0 / (rrf_k + rank)
        best[result.id] = result

    for rank, result in enumerate(bm25_results, start=1):
        scores[result.id] = scores.get(result.id, 0.0) + 1.0 / (rrf_k + rank)
        # Keep the entry with the higher original score when both lists
        # contain the same chunk.
        if result.id not in best or result.score > best[result.id].score:
            best[result.id] = result

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    merged: list[ChunkSearchResult] = []
    for chunk_id in sorted_ids:
        entry = best[chunk_id]
        merged.append(
            ChunkSearchResult(
                id=entry.id,
                physical_document_id=entry.physical_document_id,
                chunk_index=entry.chunk_index,
                page_number=entry.page_number,
                content=entry.content,
                score=scores[chunk_id],
                source=entry.source,
            )
        )
    return merged


def hybrid_search(
    client: Client,
    query: str,
    query_embedding: list[float],
    top_k: int = 5,
    rrf_k: int = 60,
    overfetch_multiplier: int = 2,
    min_similarity: float = 0.3,
    min_bm25_score: float = 0.0,
) -> list[ChunkSearchResult]:
    """Run vector and BM25 searches, then fuse results with RRF.

    Both sub-searches over-fetch by ``top_k * overfetch_multiplier`` to give
    the RRF merge more candidates.  The final list is trimmed to ``top_k``.

    The Supabase Python client is synchronous, so the two searches run
    sequentially rather than via ``asyncio.gather``.
    """
    fetch_k = top_k * overfetch_multiplier

    vector_results = similarity_search(
        client,
        query_embedding,
        top_k=fetch_k,
        min_similarity=min_similarity,
    )

    bm25_results = bm25_search(
        client,
        query,
        top_k=fetch_k,
        min_score=min_bm25_score,
    )

    merged = _merge_ranked_results(vector_results, bm25_results, rrf_k=rrf_k)
    return merged[:top_k]
