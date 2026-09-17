from unittest.mock import Mock

from app.retrieval.vector_search import similarity_search


def test_similarity_search_calls_rpc_with_correct_params() -> None:
    client = Mock()
    client.rpc.return_value.execute.return_value.data = []

    similarity_search(client, [0.1, 0.2], top_k=5, min_similarity=0.4)

    client.rpc.assert_called_once_with(
        "search_chunks_vector",
        {"query_embedding": [0.1, 0.2], "top_k": 5, "min_similarity": 0.4},
    )


def test_similarity_search_maps_rows_to_results() -> None:
    client = Mock()
    client.rpc.return_value.execute.return_value.data = [
        {
            "id": "chunk-1",
            "physical_document_id": "doc-1",
            "chunk_index": 0,
            "page_number": 3,
            "content": "test content",
            "similarity": 0.92,
        }
    ]

    results = similarity_search(client, [0.1], top_k=10)

    assert len(results) == 1
    assert results[0].id == "chunk-1"
    assert results[0].score == 0.92
    assert results[0].content == "test content"
    assert results[0].physical_document_id == "doc-1"


def test_similarity_search_handles_empty_response() -> None:
    client = Mock()
    client.rpc.return_value.execute.return_value.data = []

    results = similarity_search(client, [0.1])
    assert results == []


def test_similarity_search_handles_none_page_number() -> None:
    client = Mock()
    client.rpc.return_value.execute.return_value.data = [
        {
            "id": "chunk-1",
            "physical_document_id": "doc-1",
            "chunk_index": 0,
            "page_number": None,
            "content": "text",
            "similarity": 0.8,
        }
    ]

    results = similarity_search(client, [0.1])
    assert results[0].page_number is None
