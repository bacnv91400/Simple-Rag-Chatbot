"""Tests for GeminiChatClient SDK wrapping and error handling."""

from unittest.mock import Mock
import pytest
from google.genai.errors import ClientError

from app.generation.client import GeminiChatClient


def test_gemini_chat_client_calls_sdk_generate_content() -> None:
    mock_sdk_client = Mock()
    mock_response = Mock()
    mock_response.text = "Generated grounded answer [1]."
    mock_sdk_client.models.generate_content.return_value = mock_response

    client = GeminiChatClient(api_key="test-key", model="gemini-3.1-flash-lite", client=mock_sdk_client)
    result = client.generate("test prompt")

    assert result == "Generated grounded answer [1]."
    mock_sdk_client.models.generate_content.assert_called_once()
    call_kwargs = mock_sdk_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-3.1-flash-lite"
    assert call_kwargs["contents"] == "test prompt"


def test_gemini_chat_client_propagates_non_retryable_client_error() -> None:
    mock_sdk_client = Mock()
    error = ClientError(400, "Invalid API key", None)
    mock_sdk_client.models.generate_content.side_effect = error

    client = GeminiChatClient(api_key="bad-key", max_retries=1, client=mock_sdk_client)
    with pytest.raises(ClientError):
        client.generate("test prompt")
