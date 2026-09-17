"""Vector similarity search via the search_chunks_vector RPC."""

from __future__ import annotations

from supabase import Client

from app.retrieval.models import ChunkSearchResult


def similarity_search(
    client: Client,
    query_embedding: list[float],
    top_k: int = 10,
    min_similarity: float = 0.3,
) -> list[ChunkSearchResult]:
    """Search document chunks by cosine similarity against a query embedding.

    Calls the ``search_chunks_vector`` Postgres RPC which uses the IVFFlat
    index on ``document_chunks.embedding``.
    """
    response = client.rpc(
        "search_chunks_vector",
        {
            "query_embedding": query_embedding,
            "top_k": top_k,
            "min_similarity": min_similarity,
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
            score=float(row["similarity"]),
        )
        for row in rows
    ]
