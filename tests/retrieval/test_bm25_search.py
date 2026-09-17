from unittest.mock import Mock

from app.retrieval.bm25_search import bm25_search


def test_bm25_search_calls_rpc_with_correct_params() -> None:
    client = Mock()
    client.rpc.return_value.execute.return_value.data = []

    bm25_search(client, "machine learning", top_k=5, min_score=0.1)

    client.rpc.assert_called_once_with(
        "search_chunks_bm25",
        {"query": "machine learning", "top_k": 5, "min_score": 0.1},
    )


def test_bm25_search_maps_rows_to_results() -> None:
    client = Mock()
    client.rpc.return_value.execute.return_value.data = [
        {
            "id": "chunk-2",
            "physical_document_id": "doc-2",
            "chunk_index": 1,
            "page_number": 5,
            "content": "bm25 result",
            "score": 0.45,
        }
    ]

    results = bm25_search(client, "test query")

    assert len(results) == 1
    assert results[0].id == "chunk-2"
    assert results[0].score == 0.45
    assert results[0].content == "bm25 result"


def test_bm25_search_handles_empty_response() -> None:
    client = Mock()
    client.rpc.return_value.execute.return_value.data = []

    results = bm25_search(client, "no matches")
    assert results == []
