from unittest.mock import Mock

from app.db.client import DocumentRepository


def test_find_physical_document_returns_none_for_an_empty_result() -> None:
    client = Mock()
    client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = []

    assert DocumentRepository(client).find_physical_document("hash") is None


def test_save_successful_upload_calls_atomic_rpc() -> None:
    client = Mock()
    client.rpc.return_value.execute.return_value.data = [{"id": "upload-id"}]
    result = DocumentRepository(client).save_successful_upload(
        file_hash="hash", storage_key="hash.pdf", file_size_bytes=1, page_count=50,
        requested_filename="report.pdf", is_duplicate_content=False,
    )
    assert result == {"id": "upload-id"}
    client.rpc.assert_called_once()
