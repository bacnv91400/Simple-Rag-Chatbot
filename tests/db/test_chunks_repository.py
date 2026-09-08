from unittest.mock import Mock

from app.db.chunks_repository import ChunksRepository
from app.ingestion.chunker import ChunkRecord


def test_save_chunks_uses_idempotent_composite_upsert() -> None:
    client = Mock()
    repository = ChunksRepository(client)
    record = ChunkRecord(0, 1, "content", [0.1, 0.2])

    repository.save_chunks("document-id", [record])
    repository.save_chunks("document-id", [record])

    upsert = client.table.return_value.upsert
    assert upsert.call_count == 2
    assert upsert.call_args.kwargs["on_conflict"] == "physical_document_id,chunk_index"
    assert upsert.call_args.args[0][0]["physical_document_id"] == "document-id"


def test_claim_pending_calls_atomic_rpc() -> None:
    client = Mock()
    client.rpc.return_value.execute.return_value.data = [{"id": "doc", "storage_key": "doc.pdf"}]

    assert ChunksRepository(client).claim_pending(5) == [{"id": "doc", "storage_key": "doc.pdf"}]
    client.rpc.assert_called_once_with("claim_pending_documents", {"p_batch_size": 5})
