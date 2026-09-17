-- =====================================================================
-- Simple RAG Chatbot — Retrieval Migration
-- Target: Supabase Postgres (apply after schema.sql)
--
-- Adds full-text search support and search RPC functions for hybrid
-- retrieval (vector similarity + BM25 + Reciprocal Rank Fusion).
-- =====================================================================


-- ---------------------------------------------------------------------
-- Full-text search column (generated) + GIN index
-- ---------------------------------------------------------------------

alter table public.document_chunks
    add column if not exists fts tsvector
    generated always as (to_tsvector('english', content)) stored;

create index if not exists document_chunks_fts_idx
    on public.document_chunks using gin(fts);


-- =====================================================================
-- RPC: search_chunks_vector
--
-- Cosine similarity search against pre-computed embeddings.
-- Returns chunks ordered by descending similarity, filtered by a
-- minimum threshold.
-- =====================================================================

create or replace function public.search_chunks_vector(
    query_embedding vector(1024),
    top_k int default 10,
    min_similarity float default 0.3
)
returns table (
    id uuid,
    physical_document_id uuid,
    chunk_index int,
    page_number int,
    content text,
    similarity float
)
language plpgsql
security definer
set search_path = public, extensions
as $$
begin
    return query
    select
        dc.id,
        dc.physical_document_id,
        dc.chunk_index,
        dc.page_number,
        dc.content,
        (1 - (dc.embedding <=> query_embedding))::float as similarity
    from public.document_chunks dc
    where dc.embedding is not null
      and (1 - (dc.embedding <=> query_embedding)) >= min_similarity
    order by dc.embedding <=> query_embedding
    limit top_k;
end;
$$;


-- =====================================================================
-- RPC: search_chunks_bm25
--
-- Full-text search using websearch_to_tsquery for natural query parsing
-- and ts_rank_cd for proximity-aware ranking.
-- =====================================================================

create or replace function public.search_chunks_bm25(
    query text,
    top_k int default 10,
    min_score float default 0.0
)
returns table (
    id uuid,
    physical_document_id uuid,
    chunk_index int,
    page_number int,
    content text,
    score float
)
language plpgsql
security definer
set search_path = public, extensions
as $$
declare
    tsquery_val tsquery;
begin
    tsquery_val := websearch_to_tsquery('english', query);

    return query
    select
        dc.id,
        dc.physical_document_id,
        dc.chunk_index,
        dc.page_number,
        dc.content,
        ts_rank_cd(dc.fts, tsquery_val)::float as score
    from public.document_chunks dc
    where dc.fts @@ tsquery_val
      and ts_rank_cd(dc.fts, tsquery_val) >= min_score
    order by score desc
    limit top_k;
end;
$$;


-- =====================================================================
-- Security: grant to service_role, revoke from public/anon/authenticated
-- =====================================================================

revoke all
on function public.search_chunks_vector(vector(1024), int, float)
from public, anon, authenticated;

grant execute
on function public.search_chunks_vector(vector(1024), int, float)
to service_role;

revoke all
on function public.search_chunks_bm25(text, int, float)
from public, anon, authenticated;

grant execute
on function public.search_chunks_bm25(text, int, float)
to service_role;
