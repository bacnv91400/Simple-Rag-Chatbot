# Simple RAG Chatbot

An end-to-end Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, Supabase, Google Gemini, and IDrive e2. PDF uploads are ingested in the background while users converse in a ChatGPT-style multi-turn chat interface grounded strictly in document content with transparent source citations.

## Capabilities

```text
PDF Upload (Streamlit UI)
  -> In-memory validations (MIME, page count, SHA-256 deduplication, optional ClamAV scan)
  -> S3 storage (IDrive e2)
  -> Supabase metadata (`pending`)
  -> Background Ingestion Worker claims pending documents
  -> Lightweight extraction (PyMuPDF raster images + PyPDF text + Gemini vision captions)
  -> Docling HybridChunker + Gemini embeddings (1024-dim)
  -> Stored in Postgres `document_chunks` with `completed` status

User Question (Multi-turn Chat)
  -> Hybrid Retrieval: Vector Cosine Similarity + BM25 Full-Text Search
  -> Reciprocal Rank Fusion (RRF) -> Top-K context chunks
  -> Gemini Grounded Answer Generation (Strict refusal on insufficient context, zero hallucinations)
  -> In-text citation badges [1], [2] linked to expandable source cards
  -> Chat persistence: `chat_sessions`, `messages`, and `message_sources` in Supabase
  -> Browser refresh restores previous conversations and documents
```

### Key Features

* **ChatGPT-style Interface**: Left sidebar with `+ New Chat`, full conversation history, document status monitor, and expandable processing logs.
* **Hybrid Retrieval (Vector + BM25 + RRF)**: High-recall hybrid search combining dense semantic similarity and keyword BM25 queries merged with Reciprocal Rank Fusion.
* **Grounded Generation & Citations**: Answers are strictly grounded in retrieved chunks with citations like `[1]`, `[2]`. If context is insufficient, the system safely refuses rather than hallucinating.
* **Expandable Source Inspector**: Each source chunk can be expanded to inspect the raw excerpt, document filename, page number, and RRF score (never misleading confidence scores).
* **Multi-Turn Chat Persistence**: Conversations, messages, and chunk source mappings persist in Supabase (`chat_sessions`, `messages`, `message_sources`). Automatic session titling from first questions.
* **Live Document Status**: Polling via `@st.fragment` displays document page counts and states (`Queued`, `Processing`, `Ready`, `Failed`) with one-click `Retry`.

---

## Architecture

| Area | Responsibility |
| --- | --- |
| **Streamlit UI** (`app/ui/`) | Modern layout: sidebar sessions, live document monitoring, chat messages, suggestion pills, expandable citations, and sticky input. |
| **Chat & Source Persistence** (`app/db/chat_repository.py`) | Supabase persistence for `chat_sessions`, `messages`, `message_sources`, and document queries. |
| **Grounded Generation** (`app/generation/`) | Strict grounding prompt engineering, Gemini chat client with exponential backoff retries, and citation index parser. |
| **Hybrid Retrieval** (`app/retrieval/`) | Dense vector similarity, BM25 full-text search, and RRF rank fusion. |
| **Ingestion Pipeline** (`app/ingestion/`) | Validation, deduplication, S3 upload, and metadata persistence. |
| **Background Worker** (`worker/`) | FIFO claim queue, image extraction, captioning, chunking, and embedding. |

---

## Configuration

Copy `.env.example` to `.env` and configure:

| Variable | Purpose | Default |
| --- | --- | --- |
| `APP_NAME` | Streamlit title | `Simple RAG Chatbot` |
| `APP_ENV` | Environment name | `development` |
| `SUPABASE_URL` | Supabase project URL | Required |
| `SUPABASE_SERVICE_ROLE_KEY` | Server-side Supabase key | Required |
| `IDRIVE_E2_ENDPOINT` / `IDRIVE_E2_BUCKET` | IDrive e2 S3 settings | Required |
| `GOOGLE_API_KEY` | Google GenAI API key | Required |
| `GEMINI_CHAT_MODEL` | Gemini LLM for answer generation | `gemini-3.1-flash-lite` |
| `GEMINI_CAPTION_MODEL` | Gemini vision model for PDF images | `gemini-3.1-flash-lite` |
| `GEMINI_EMBED_MODEL` | Gemini embedding model | `gemini-embedding-001` |
| `RETRIEVAL_TOP_K` | Number of top chunks to return | `5` |
| `RETRIEVAL_SIMILARITY_THRESHOLD` | Cosine similarity floor | `0.3` |
| `RETRIEVAL_RRF_K` | RRF constant parameter | `60` |
| `RETRIEVAL_OVERFETCH_MULTIPLIER` | Retrieval overfetch factor | `2` |

---

## Quickstart

### Prerequisites
- Python 3.12
- [uv](https://docs.astral.sh/uv/) or standard virtualenv
- Supabase project with `db/schema.sql` and `db/retrieval.sql` applied

### Local Development

1. Setup environment:
   ```powershell
   Copy-Item .env.example .env
   # Fill in SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, GOOGLE_API_KEY, and IDrive credentials
   ```

2. Run tests:
   ```powershell
   & "venv\Scripts\Activate.ps1"; uv run pytest
   ```

3. Launch application:
   ```powershell
   & "venv\Scripts\Activate.ps1"; uv run streamlit run app/main.py
   ```

### Docker Stack

Run the full stack (Streamlit UI, Ingestion Worker, and ClamAV):
```powershell
docker compose up --build
```
Access the application at `http://localhost:8501`.
