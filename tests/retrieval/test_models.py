from app.retrieval.models import ChunkSearchResult


def test_chunk_search_result_defaults() -> None:
    result = ChunkSearchResult(
        id="chunk-1",
        physical_document_id="doc-1",
        chunk_index=0,
        page_number=1,
        content="hello",
        score=0.9,
    )

    assert result.source == ""
    assert result.page_number == 1
    assert result.score == 0.9


def test_chunk_search_result_accepts_none_page() -> None:
    result = ChunkSearchResult(
        id="chunk-1",
        physical_document_id="doc-1",
        chunk_index=0,
        page_number=None,
        content="hello",
        score=0.5,
    )

    assert result.page_number is None
