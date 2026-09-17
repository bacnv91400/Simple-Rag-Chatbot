"""BM25 full-text search via the search_chunks_bm25 RPC."""

from __future__ import annotations

from supabase import Client

from app.retrieval.models import ChunkSearchResult


def bm25_search(
    client: Client,
    query: str,
    top_k: int = 10,
    min_score: float = 0.0,
) -> list[ChunkSearchResult]:
    """Search document chunks using Postgres full-text search (ts_rank_cd).

    Calls the ``search_chunks_bm25`` Postgres RPC which uses
    ``websearch_to_tsquery`` and the GIN index on ``document_chunks.fts``.
    """
    response = client.rpc(
        "search_chunks_bm25",
        {
            "query": query,
            "top_k": top_k,
            "min_score": min_score,
        },
    ).execute()

    rows = getattr(response, "data", None) or []
    return [
        ChunkSearchResult(
            id=row["id"],
            physical_document_id=row["physical_document_id"],
            chunk_index=row["chunk_index"],
            page_number=row.get("page_number"),
            content=row["content"],
            score=float(row["score"]),
        )
        for row in rows
    ]
