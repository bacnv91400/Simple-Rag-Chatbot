from app.retrieval.hybrid_search import _merge_ranked_results
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


def test_rrf_disjoint_lists() -> None:
    """Non-overlapping results from each source get independent RRF scores."""
    vector = [_make_result("v1"), _make_result("v2")]
    bm25 = [_make_result("b1"), _make_result("b2")]

    merged = _merge_ranked_results(vector, bm25, rrf_k=60)

    ids = [r.id for r in merged]
    assert len(ids) == 4
    # First-ranked items from each list should appear before second-ranked
    assert ids.index("v1") < ids.index("v2")
    assert ids.index("b1") < ids.index("b2")


def test_rrf_overlapping_results_boosted() -> None:
    """A chunk appearing in both lists gets a higher fused score."""
    shared = _make_result("shared", score=0.8)
    vector = [shared, _make_result("v-only")]
    bm25 = [_make_result("shared", score=0.6), _make_result("b-only")]

    merged = _merge_ranked_results(vector, bm25, rrf_k=60)

    assert merged[0].id == "shared"
    # Score should be sum of both reciprocal ranks
    expected_score = 1.0 / (60 + 1) + 1.0 / (60 + 1)
    assert abs(merged[0].score - expected_score) < 1e-9


def test_rrf_single_source_vector_only() -> None:
    """When BM25 returns nothing, vector results are still returned."""
    vector = [_make_result("v1"), _make_result("v2")]
    merged = _merge_ranked_results(vector, [], rrf_k=60)

    assert [r.id for r in merged] == ["v1", "v2"]


def test_rrf_single_source_bm25_only() -> None:
    """When vector returns nothing, BM25 results are still returned."""
    bm25 = [_make_result("b1"), _make_result("b2")]
    merged = _merge_ranked_results([], bm25, rrf_k=60)

    assert [r.id for r in merged] == ["b1", "b2"]


def test_rrf_empty_inputs() -> None:
    """Empty inputs from both sides produce an empty result."""
    assert _merge_ranked_results([], [], rrf_k=60) == []


def test_rrf_preserves_content() -> None:
    """Merged results carry through the original content field."""
    vector = [_make_result("v1")]
    merged = _merge_ranked_results(vector, [], rrf_k=60)

    assert merged[0].content == "content-v1"


def test_rrf_score_decreases_with_rank() -> None:
    """Lower-ranked items receive smaller RRF scores."""
    vector = [_make_result(f"v{i}") for i in range(5)]
    merged = _merge_ranked_results(vector, [], rrf_k=60)

    scores = [r.score for r in merged]
    assert scores == sorted(scores, reverse=True)
