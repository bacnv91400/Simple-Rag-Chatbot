"""Structure-aware chunking with token overlap and provenance metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class Chunk:
    index: int
    text: str
    page_start: int | None = None
    page_end: int | None = None
    section: str = ""
    content_type: str = "text"
    token_count: int = 0
    overlap_tokens: int = 0
    overlap_from_chunk_index: int | None = None
    embedding: list[float] | None = None

    @property
    def chunk_index(self) -> int:
        return self.index

    @property
    def page_number(self) -> int | None:
        return self.page_start

    @property
    def content(self) -> str:
        return self.text


class ChunkRecord(Chunk):
    """Backward-compatible constructor for legacy ChunkRecord(index, page, content, embedding)."""

    def __init__(
        self,
        chunk_index: int,
        page_number: int | None,
        content: str,
        embedding: list[float] | None = None,
        page_end: int | None = None,
        section: str = "",
        content_type: str = "text",
        token_count: int = 0,
        overlap_tokens: int = 0,
        overlap_from_chunk_index: int | None = None,
    ) -> None:
        super().__init__(
            index=chunk_index,
            text=content,
            page_start=page_number,
            page_end=page_end if page_end is not None else page_number,
            section=section,
            content_type=content_type,
            token_count=token_count,
            overlap_tokens=overlap_tokens,
            overlap_from_chunk_index=overlap_from_chunk_index,
            embedding=embedding,
        )


def clean_chunk_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def contextualize_chunk(chunk: Any, chunker: Any = None) -> str:
    if chunker is not None and hasattr(chunker, "contextualize"):
        try:
            return chunker.contextualize(chunk)
        except Exception:
            pass
    return getattr(chunk, "text", str(chunk))


def extract_chunk_headings(chunk: Any) -> str:
    try:
        headings = getattr(chunk.meta, "headings", None)
        if headings:
            return " > ".join(str(x).strip() for x in headings if str(x).strip())
    except Exception:
        pass
    return ""


def extract_chunk_pages(chunk: Any) -> list[int]:
    pages = set()
    try:
        meta = getattr(chunk, "meta", None)
        if meta is not None:
            for item in getattr(meta, "doc_items", []) or []:
                for prov in getattr(item, "prov", []) or []:
                    page_no = getattr(prov, "page_no", None)
                    if page_no is not None:
                        pages.add(int(page_no))
    except Exception:
        pass
    return sorted(pages)


def infer_content_type(text: str) -> str:
    return "image" if text.startswith("[Visual content on page") else "text"


def take_last_tokens(text: str, n_tokens: int, hf_tokenizer: Any) -> str:
    if n_tokens <= 0 or not text:
        return ""
    encoded = hf_tokenizer.encode(text, add_special_tokens=False)
    if len(encoded) <= n_tokens:
        return text.strip()
    tail_ids = encoded[-n_tokens:]
    return hf_tokenizer.decode(tail_ids, skip_special_tokens=True).strip()


def build_chunk_record(index: int, raw_chunk: Any, chunker: Any, docling_tokenizer: Any) -> Chunk:
    raw_text = contextualize_chunk(raw_chunk, chunker)
    text = clean_chunk_text(raw_text)
    pages = extract_chunk_pages(raw_chunk)
    section = extract_chunk_headings(raw_chunk)
    c_type = infer_content_type(text)
    t_count = docling_tokenizer.count_tokens(text=text)
    return Chunk(
        index=index,
        text=text,
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        section=section,
        content_type=c_type,
        token_count=t_count,
        overlap_tokens=0,
        overlap_from_chunk_index=None,
    )


def apply_token_overlap(
    chunks: list[Chunk],
    overlap_tokens: int,
    max_tokens: int,
    docling_tokenizer: Any,
    hf_tokenizer: Any,
) -> list[Chunk]:
    """Add sliding-window token overlap from previous chunk, respecting max_tokens budget."""
    if overlap_tokens <= 0 or len(chunks) <= 1:
        return chunks

    for i in range(1, len(chunks)):
        chunk = chunks[i]
        current_tokens = docling_tokenizer.count_tokens(text=chunk.text)
        budget = max_tokens - current_tokens
        if budget <= 0:
            continue  # already at cap; skip overlap rather than exceed it
        effective_overlap = min(overlap_tokens, budget)
        tail = take_last_tokens(chunks[i - 1].text, effective_overlap, hf_tokenizer)
        if tail:
            chunk.text = clean_chunk_text(tail + "\n\n" + chunk.text)
            chunk.overlap_tokens = effective_overlap
            chunk.overlap_from_chunk_index = i - 1
            chunk.token_count = docling_tokenizer.count_tokens(text=chunk.text)

    return chunks


# Note: gemini-embedding-001 has no public HF-compatible tokenizer,
# so whatever HF tokenizer is configured here is only an approximation
# of Gemini's real token count. CHUNK_MAX_TOKENS (512) leaves comfortable
# margin under Gemini's real 2048-token limit.
def chunk_document(
    doc: Any,
    tokenizer_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    """Chunk a DoclingDocument using HybridChunker and post-chunk overlap."""
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from transformers import AutoTokenizer

    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    docling_tokenizer = HuggingFaceTokenizer(tokenizer=hf_tokenizer, max_tokens=max_tokens)
    chunker = HybridChunker(tokenizer=docling_tokenizer, merge_peers=True)
    raw_chunks = list(chunker.chunk(dl_doc=doc))

    chunks = [build_chunk_record(i, c, chunker, docling_tokenizer) for i, c in enumerate(raw_chunks)]
    return apply_token_overlap(chunks, overlap_tokens, max_tokens, docling_tokenizer, hf_tokenizer)
