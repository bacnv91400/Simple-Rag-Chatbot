"""Gemini chat client using the Google GenAI SDK with retry handling."""

from __future__ import annotations

import logging
from typing import Any

from google import genai
from google.genai import types as genai_types
from google.genai.errors import ClientError, ServerError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.generation.prompts import SYSTEM_INSTRUCTION

logger = logging.getLogger(__name__)


def _retryable_error(error: BaseException) -> bool:
    """Retry on 5xx server errors and 429 rate limit errors."""
    if isinstance(error, ServerError):
        return True
    if isinstance(error, ClientError):
        return getattr(error, "code", None) == 429
    return False


class GeminiChatClient:
    """Client for generating grounded chat answers via Google Gemini."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.1-flash-lite",
        max_retries: int = 3,
        retry_base_delay: float = 1.5,
        client: Any = None,
    ) -> None:
        self.client = client or genai.Client(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

    def _make_retrying_caller(self):
        @retry(
            retry=retry_if_exception(_retryable_error),
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=self.retry_base_delay, min=self.retry_base_delay, max=20),
            reraise=True,
        )
        def _call(prompt: str, system_instruction: str) -> str:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                ),
            )
            return response.text or ""

        return _call

    def generate(self, prompt: str, system_instruction: str = SYSTEM_INSTRUCTION) -> str:
        """Call Gemini to generate a response from prompt and system instruction."""
        caller = self._make_retrying_caller()
        try:
            return caller(prompt, system_instruction)
        except Exception as error:
            logger.error("Gemini text generation failed: %s", error)
            raise
