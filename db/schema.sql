-- =====================================================================
-- Simple RAG Chatbot — Database Schema
-- Target: Supabase Postgres
-- =====================================================================

-- ---------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------

create extension if not exists pgcrypto;
create extension if not exists vector;


-- =====================================================================
-- physical_documents
-- One row per unique file content
-- =====================================================================

create table if not exists public.physical_documents (
    id                  uuid primary key default gen_random_uuid(),
    file_hash           text not null unique,
    storage_key         text not null unique,
    file_size_bytes     bigint not null,
    page_count          integer not null,

    processing_status   text not null default 'pending'
                        check (
                            processing_status in (
                                'pending',
                                'processing',
                                'completed',
                                'failed'
                            )
                        ),

    processing_error    text,

    created_at          timestamptz not null default now(),
    updated_at          timestamptz not null default now()
);


-- =====================================================================
-- document_uploads
-- One row per successful upload event
-- =====================================================================

create table if not exists public.document_uploads (
    id                   uuid primary key default gen_random_uuid(),

    physical_document_id uuid not null
                         references public.physical_documents(id),

    display_filename     text not null unique,

    is_duplicate_content boolean not null default false,

    created_at            timestamptz not null default now()
);


create index if not exists document_uploads_physical_doc_idx
    on public.document_uploads (physical_document_id);


-- =====================================================================
-- document_chunks
-- Retrieval units
-- =====================================================================

create table if not exists public.document_chunks (
    id                   uuid primary key default gen_random_uuid(),

    physical_document_id uuid not null
                         references public.physical_documents(id)
                         on delete cascade,

    chunk_index          integer not null,

    page_number          integer,

    content              text not null,

    embedding            vector(1024),

    created_at            timestamptz not null default now(),

    unique (physical_document_id, chunk_index)
);


create index if not exists document_chunks_physical_doc_idx
    on public.document_chunks (physical_document_id);


-- ANN index for cosine similarity search
create index if not exists document_chunks_embedding_idx
    on public.document_chunks
    using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);


-- =====================================================================
-- chat_sessions
-- =====================================================================

create table if not exists public.chat_sessions (
    id          uuid primary key default gen_random_uuid(),
    title       text,
    created_at  timestamptz not null default now()
);


-- =====================================================================
-- messages
-- =====================================================================

create table if not exists public.messages (
    id               uuid primary key default gen_random_uuid(),

    chat_session_id  uuid not null
                     references public.chat_sessions(id)
                     on delete cascade,

    role             text not null
                     check (role in ('user', 'assistant', 'system')),

    content          text not null,

    created_at       timestamptz not null default now()
);


create index if not exists messages_session_idx
    on public.messages (chat_session_id, created_at);


-- =====================================================================
-- message_sources
-- =====================================================================

create table if not exists public.message_sources (
    id              uuid primary key default gen_random_uuid(),

    message_id      uuid not null
                    references public.messages(id)
                    on delete cascade,

    chunk_id        uuid not null
                    references public.document_chunks(id),

    relevance_score float
);


create index if not exists message_sources_message_idx
    on public.message_sources (message_id);


-- =====================================================================
-- RPC: save_successful_upload
--
-- Atomically:
--   1. Creates/reuses physical_documents row
--   2. Generates unique display filename
--   3. Creates document_uploads row
-- =====================================================================

create or replace function public.save_successful_upload(
    p_file_hash text,
    p_storage_key text,
    p_file_size_bytes bigint,
    p_page_count integer,
    p_requested_filename text,
    p_is_duplicate_content boolean
)
returns table (
    id uuid,
    physical_document_id uuid,
    display_filename text,
    is_duplicate_content boolean,
    created_at timestamptz
)
language plpgsql
security definer
set search_path = public
as $$
declare
    v_physical_id uuid;
    v_filename text := p_requested_filename;
    v_base text;
    v_extension text;
    v_suffix integer := 1;
begin

    -- -------------------------------------------------------------
    -- Insert physical document.
    -- If the file hash already exists, reuse the existing row.
    -- -------------------------------------------------------------

    insert into public.physical_documents (
        file_hash,
        storage_key,
        file_size_bytes,
        page_count
    )
    values (
        p_file_hash,
        p_storage_key,
        p_file_size_bytes,
        p_page_count
    )
    on conflict (file_hash)
    do update
        set file_hash = excluded.file_hash
    returning public.physical_documents.id
    into v_physical_id;


    -- -------------------------------------------------------------
    -- Split filename into base + extension
    -- -------------------------------------------------------------

    v_extension :=
        coalesce(
            substring(v_filename from '(\.[^.]+)$'),
            ''
        );

    v_base :=
        left(
            v_filename,
            length(v_filename) - length(v_extension)
        );


    -- -------------------------------------------------------------
    -- Avoid duplicate display filenames
    -- example:
    --   document.pdf
    --   document (1).pdf
    --   document (2).pdf
    -- -------------------------------------------------------------

    while exists (
        select 1
        from public.document_uploads d
        where d.display_filename = v_filename
    )
    loop

        v_filename :=
            v_base
            || ' ('
            || v_suffix
            || ')'
            || v_extension;

        v_suffix := v_suffix + 1;

    end loop;


    -- -------------------------------------------------------------
    -- Create successful upload record
    -- -------------------------------------------------------------

    return query
    insert into public.document_uploads (
        physical_document_id,
        display_filename,
        is_duplicate_content
    )
    values (
        v_physical_id,
        v_filename,
        p_is_duplicate_content
    )
    returning
        public.document_uploads.id,
        public.document_uploads.physical_document_id,
        public.document_uploads.display_filename,
        public.document_uploads.is_duplicate_content,
        public.document_uploads.created_at;

end;
$$;


-- =====================================================================
-- ROW LEVEL SECURITY
--
-- These tables are not exposed to anon/publishable client access.
-- The application uses the server-side service_role key.
-- =====================================================================

alter table public.physical_documents enable row level security;
alter table public.document_uploads enable row level security;
alter table public.document_chunks enable row level security;
alter table public.chat_sessions enable row level security;
alter table public.messages enable row level security;
alter table public.message_sources enable row level security;


-- =====================================================================
-- SERVICE ROLE TABLE PRIVILEGES
--
-- IMPORTANT:
-- API authentication and PostgreSQL authorization are separate.
--
-- The service_role API key authenticates the server request as the
-- service_role role. PostgreSQL still needs table privileges.
-- =====================================================================

grant select, insert, update, delete
on table public.physical_documents
to service_role;

grant select, insert, update, delete
on table public.document_uploads
to service_role;

grant select, insert, update, delete
on table public.document_chunks
to service_role;

grant select, insert, update, delete
on table public.chat_sessions
to service_role;

grant select, insert, update, delete
on table public.messages
to service_role;

grant select, insert, update, delete
on table public.message_sources
to service_role;


-- =====================================================================
-- SERVICE ROLE RPC PRIVILEGE
-- =====================================================================

grant execute
on function public.save_successful_upload(
    text,
    text,
    bigint,
    integer,
    text,
    boolean
)
to service_role;


-- =====================================================================
-- OPTIONAL: explicitly remove direct table access from anon/authenticated
--
-- Since this is a server-only, single-tenant application, these roles
-- should not be able to access the tables through the Data API.
-- =====================================================================

revoke all
on table public.physical_documents
from anon, authenticated;

revoke all
on table public.document_uploads
from anon, authenticated;

revoke all
on table public.document_chunks
from anon, authenticated;

revoke all
on table public.chat_sessions
from anon, authenticated;

revoke all
on table public.messages
from anon, authenticated;

revoke all
on table public.message_sources
from anon, authenticated;


-- =====================================================================
-- RPC SECURITY
--
-- The function is intended for the server-side service_role only.
-- =====================================================================

revoke all
on function public.save_successful_upload(
    text,
    text,
    bigint,
    integer,
    text,
    boolean
)
from public;

revoke all
on function public.save_successful_upload(
    text,
    text,
    bigint,
    integer,
    text,
    boolean
)
from anon, authenticated;

grant execute
on function public.save_successful_upload(
    text,
    text,
    bigint,
    integer,
    text,
    boolean
)
to service_role;