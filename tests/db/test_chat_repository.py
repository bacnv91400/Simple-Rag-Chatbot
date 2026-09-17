"""Tests for ChatRepository persistence and retrieval."""

from unittest.mock import Mock
import pytest

from app.db.chat_repository import ChatRepository
from app.retrieval.models import ChunkSearchResult


def test_create_session() -> None:
    client = Mock()
    client.table.return_value.insert.return_value.execute.return_value.data = [
        {"id": "session-1", "title": "New Chat", "created_at": "2026-09-13T10:00:00Z"}
    ]
    repo = ChatRepository(client)
    session = repo.create_session("New Chat")

    assert session["id"] == "session-1"
    client.table.assert_called_with("chat_sessions")
    client.table.return_value.insert.assert_called_with({"title": "New Chat"})


def test_list_sessions() -> None:
    client = Mock()
    client.table.return_value.select.return_value.order.return_value.execute.return_value.data = [
        {"id": "s1", "title": "Chat 1"},
        {"id": "s2", "title": "Chat 2"},
    ]
    repo = ChatRepository(client)
    sessions = repo.list_sessions()

    assert len(sessions) == 2
    assert sessions[0]["id"] == "s1"
    client.table.assert_called_with("chat_sessions")


def test_update_and_delete_session() -> None:
    client = Mock()
    repo = ChatRepository(client)

    repo.update_session_title("s1", "Updated Title")
    client.table.return_value.update.assert_called_with({"title": "Updated Title"})

    repo.delete_session("s1")
    client.table.return_value.delete.return_value.eq.assert_called_with("id", "s1")


def test_add_message_and_sources() -> None:
    client = Mock()
    client.table.return_value.insert.return_value.execute.return_value.data = [
        {"id": "msg-1", "chat_session_id": "s1", "role": "assistant", "content": "Answer [1]."}
    ]
    repo = ChatRepository(client)

    msg = repo.add_message("s1", "assistant", "Answer [1].")
    assert msg["id"] == "msg-1"

    chunks = [
        ChunkSearchResult(
            id="chunk-abc",
            physical_document_id="p1",
            chunk_index=0,
            page_number=3,
            content="Context",
            score=0.035,
            source="Doc.pdf",
        )
    ]
    repo.add_message_sources("msg-1", chunks)

    client.table.assert_called_with("message_sources")
    insert_call = client.table.return_value.insert.call_args[0][0]
    assert insert_call[0]["message_id"] == "msg-1"
    assert insert_call[0]["chunk_id"] == "chunk-abc"
    assert insert_call[0]["relevance_score"] == 0.035


def test_get_messages_with_sources_resolves_details() -> None:
    client = Mock()
    # Mock messages table
    messages_mock = Mock()
    messages_mock.data = [
        {"id": "m1", "chat_session_id": "s1", "role": "user", "content": "Question"},
        {"id": "m2", "chat_session_id": "s1", "role": "assistant", "content": "Answer [1]"},
    ]

    # Mock message_sources table
    sources_mock = Mock()
    sources_mock.data = [
        {"id": "ms1", "message_id": "m2", "chunk_id": "c1", "relevance_score": 0.04}
    ]

    # Mock document_chunks table
    chunks_mock = Mock()
    chunks_mock.data = [
        {"id": "c1", "physical_document_id": "p1", "page_number": 5, "content": "Context snippet"}
    ]

    # Mock document_uploads table
    uploads_mock = Mock()
    uploads_mock.data = [
        {"physical_document_id": "p1", "display_filename": "Report.pdf"}
    ]

    def table_side_effect(table_name: str):
        mock_t = Mock()
        if table_name == "messages":
            mock_t.select.return_value.eq.return_value.order.return_value.execute.return_value = messages_mock
        elif table_name == "message_sources":
            mock_t.select.return_value.in_.return_value.execute.return_value = sources_mock
        elif table_name == "document_chunks":
            mock_t.select.return_value.in_.return_value.execute.return_value = chunks_mock
        elif table_name == "document_uploads":
            mock_t.select.return_value.in_.return_value.execute.return_value = uploads_mock
        return mock_t

    client.table.side_effect = table_side_effect

    repo = ChatRepository(client)
    msgs = repo.get_messages_with_sources("s1")

    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["sources"] == []

    assert msgs[1]["role"] == "assistant"
    assert len(msgs[1]["sources"]) == 1
    src = msgs[1]["sources"][0]
    assert src["chunk_id"] == "c1"
    assert src["source"] == "Report.pdf"
    assert src["page_number"] == 5
    assert src["score"] == 0.04


def test_get_all_documents_joins_uploads_and_physical_docs() -> None:
    client = Mock()
    uploads_mock = Mock()
    uploads_mock.data = [
        {"id": "u1", "physical_document_id": "p1", "display_filename": "Annual Report.pdf", "created_at": "2026-09-13T10:00:00Z"}
    ]
    pdocs_mock = Mock()
    pdocs_mock.data = [
        {"id": "p1", "page_count": 124, "processing_status": "completed", "processing_error": None}
    ]

    def table_side_effect(table_name: str):
        mock_t = Mock()
        if table_name == "document_uploads":
            mock_t.select.return_value.order.return_value.execute.return_value = uploads_mock
        elif table_name == "physical_documents":
            mock_t.select.return_value.in_.return_value.execute.return_value = pdocs_mock
        return mock_t

    client.table.side_effect = table_side_effect

    repo = ChatRepository(client)
    docs = repo.get_all_documents()

    assert len(docs) == 1
    assert docs[0]["display_filename"] == "Annual Report.pdf"
    assert docs[0]["page_count"] == 124
    assert docs[0]["processing_status"] == "completed"

