"""Gemini API embedding client with rate limiting and retry handling."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from google import genai
from google.genai import types as genai_types
from google.genai.errors import ClientError, ServerError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config.settings import Settings

logger = logging.getLogger(__name__)


class MinIntervalRateLimiter:
    """Thread-safe pacing limiter ensuring minimum delay between successive calls."""

    def __init__(self, min_interval_seconds: float) -> None:
        self._min_interval = max(0.0, min_interval_seconds)
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            remaining = self._min_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
            self._last_call = time.monotonic()


def l2_normalize_single(vector: Sequence[float] | np.ndarray) -> list[float]:
    """Normalize vector to unit length (L2 norm = 1.0)."""
    arr = np.array(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr.tolist()
    return (arr / norm).tolist()


def _retryable_error(error: BaseException) -> bool:
    """Retry on 5xx server errors and 429 rate limit errors."""
    if isinstance(error, ServerError):
        return True
    if isinstance(error, ClientError):
        return getattr(error, "code", None) == 429
    return False


class GeminiEmbedder:
    """Embeds single text strings using the Google GenAI SDK with rate limiting and retries."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        output_dim: int = 1024,
        rate_limiter: MinIntervalRateLimiter | None = None,
        max_retries: int = 4,
        retry_base_delay: float = 2.0,
        client: Any = None,
    ) -> None:
        self.client = client or genai.Client(api_key=api_key)
        self.model = model
        self.output_dim = output_dim
        self._rate_limiter = rate_limiter or MinIntervalRateLimiter(1.0)
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

    def _make_retrying_caller(self):
        @retry(
            retry=retry_if_exception(_retryable_error),
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=self.retry_base_delay, min=self.retry_base_delay, max=30),
            reraise=True,
        )
        def _call(text: str, task_type: str):
            return self.client.models.embed_content(
                model=self.model,
                contents=text,
                config=genai_types.EmbedContentConfig(
                    output_dimensionality=self.output_dim,
                    task_type=task_type,
                ),
            )

        return _call

    def embed_one(
        self,
        text: str,
        task_type: Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"] = "RETRIEVAL_DOCUMENT",
    ) -> list[float]:
        """Embed a single text string and return an L2-normalized vector."""
        self._rate_limiter.wait()
        caller = self._make_retrying_caller()
        response = caller(text, task_type)
        if not response.embeddings:
            raise ValueError("Embedding API returned no embeddings")
        vector = response.embeddings[0].values
        return l2_normalize_single(vector)


def embed_texts(
    texts: Sequence[str],
    settings: Settings,
    rate_limiter: MinIntervalRateLimiter | None = None,
) -> list[list[float]]:
    """Embed multiple texts sequentially using GeminiEmbedder."""
    embedder = GeminiEmbedder(
        api_key=settings.google_api_key or "",
        model=settings.gemini_embed_model,
        output_dim=settings.embed_output_dim,
        rate_limiter=rate_limiter or MinIntervalRateLimiter(settings.embed_min_seconds_between_requests),
        max_retries=settings.embed_max_retries,
        retry_base_delay=settings.embed_retry_base_delay,
    )
    return [embedder.embed_one(text, "RETRIEVAL_DOCUMENT") for text in texts]