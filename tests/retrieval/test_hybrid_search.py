from unittest.mock import Mock, patch

from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.models import ChunkSearchResult


def _make_result(chunk_id: str, score: float = 0.5) -> ChunkSearchResult:
    return ChunkSearchResult(
        id=chunk_id,
        physical_document_id="doc-1",
        chunk_index=0,
        page_number=1,
        content=f"content-{chunk_id}",
        score=score,
    )


@patch("app.retrieval.hybrid_search.bm25_search")
@patch("app.retrieval.hybrid_search.similarity_search")
def test_hybrid_search_overfetches(mock_vector, mock_bm25) -> None:
    """Each sub-search receives top_k * overfetch_multiplier."""
    mock_vector.return_value = []
    mock_bm25.return_value = []
    client = Mock()

    hybrid_search(
        client, "query", [0.1], top_k=5, overfetch_multiplier=3,
    )

    _, vector_kwargs = mock_vector.call_args
    _, bm25_kwargs = mock_bm25.call_args
    assert vector_kwargs["top_k"] == 15
    assert bm25_kwargs["top_k"] == 15


@patch("app.retrieval.hybrid_search.bm25_search")
@patch("app.retrieval.hybrid_search.similarity_search")
def test_hybrid_search_trims_to_top_k(mock_vector, mock_bm25) -> None:
    """Final result list is trimmed to top_k."""
    mock_vector.return_value = [_make_result(f"v{i}") for i in range(10)]
    mock_bm25.return_value = [_make_result(f"b{i}") for i in range(10)]
    client = Mock()

    results = hybrid_search(client, "query", [0.1], top_k=3)

    assert len(results) == 3


@patch("app.retrieval.hybrid_search.bm25_search")
@patch("app.retrieval.hybrid_search.similarity_search")
def test_hybrid_search_returns_fused_order(mock_vector, mock_bm25) -> None:
    """Chunks appearing in both lists are ranked higher than single-source."""
    shared = _make_result("shared", score=0.9)
    mock_vector.return_value = [shared, _make_result("v-only")]
    mock_bm25.return_value = [_make_result("shared", score=0.7), _make_result("b-only")]
    client = Mock()

    results = hybrid_search(client, "query", [0.1], top_k=10)

    assert results[0].id == "shared"


@patch("app.retrieval.hybrid_search.bm25_search")
@patch("app.retrieval.hybrid_search.similarity_search")
def test_hybrid_search_passes_thresholds(mock_vector, mock_bm25) -> None:
    """Similarity and BM25 thresholds are forwarded to sub-searches."""
    mock_vector.return_value = []
    mock_bm25.return_value = []
    client = Mock()

    hybrid_search(
        client, "query", [0.1],
        min_similarity=0.5, min_bm25_score=0.2,
    )

    _, vector_kwargs = mock_vector.call_args
    _, bm25_kwargs = mock_bm25.call_args
    assert vector_kwargs["min_similarity"] == 0.5
    assert bm25_kwargs["min_score"] == 0.2


@patch("app.retrieval.hybrid_search.bm25_search")
@patch("app.retrieval.hybrid_search.similarity_search")
def test_hybrid_search_empty_both(mock_vector, mock_bm25) -> None:
    """Empty results from both sources produce an empty result."""
    mock_vector.return_value = []
    mock_bm25.return_value = []
    client = Mock()

    results = hybrid_search(client, "query", [0.1])
    assert results == []
