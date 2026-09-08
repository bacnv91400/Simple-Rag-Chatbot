"""Tests for Gemini embedder and MinIntervalRateLimiter."""

import math
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

from app.ingestion.embedder import (
    GeminiEmbedder,
    MinIntervalRateLimiter,
    l2_normalize_single,
)


def test_embed_one_passes_single_string() -> None:
    mock_client = Mock()
    mock_client.models.embed_content.return_value = SimpleNamespace(
        embeddings=[SimpleNamespace(values=[3.0, 4.0])]
    )

    embedder = GeminiEmbedder(
        api_key="test-key",
        model="gemini-embedding-001",
        output_dim=2,
        rate_limiter=MinIntervalRateLimiter(0.0),
        client=mock_client,
    )

    result = embedder.embed_one("hello world", task_type="RETRIEVAL_DOCUMENT")

    mock_client.models.embed_content.assert_called_once()
    call_kwargs = mock_client.models.embed_content.call_args.kwargs
    # CRITICAL: contents must be a str, not a list
    assert isinstance(call_kwargs["contents"], str)
    assert call_kwargs["contents"] == "hello world"
    assert call_kwargs["model"] == "gemini-embedding-001"
    assert call_kwargs["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert call_kwargs["config"].output_dimensionality == 2

    # Verify L2 normalization: [3, 4] -> norm is 5 -> [0.6, 0.8]
    assert math.isclose(result[0], 0.6, abs_tol=1e-5)
    assert math.isclose(result[1], 0.8, abs_tol=1e-5)
    norm = math.sqrt(sum(x * x for x in result))
    assert math.isclose(norm, 1.0, abs_tol=1e-5)


def test_embed_one_l2_normalizes() -> None:
    unnormalized = [1.0, 2.0, 3.0, 4.0]
    normalized = l2_normalize_single(unnormalized)
    norm = math.sqrt(sum(x * x for x in normalized))
    assert math.isclose(norm, 1.0, abs_tol=1e-5)


def test_rate_limiter_concurrent_calls_respect_spacing() -> None:
    interval = 0.05  # 50ms
    limiter = MinIntervalRateLimiter(min_interval_seconds=interval)

    timestamps: list[float] = []
    lock = threading.Lock()

    def worker():
        limiter.wait()
        with lock:
            timestamps.append(time.monotonic())

    threads = [threading.Thread(target=worker) for _ in range(4)]
    start = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    timestamps.sort()
    # Ensure there are 4 timestamps
    assert len(timestamps) == 4
    # Check that each adjacent interval is at least close to interval (allowing small timing jitter)
    for i in range(len(timestamps) - 1):
        diff = timestamps[i + 1] - timestamps[i]
        assert diff >= interval * 0.8, f"Call {i} to {i+1} spacing was {diff}s, expected >={interval}s"
