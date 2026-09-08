"""Polling entry point for the stateless background ingestion worker."""

from __future__ import annotations

import asyncio
import logging
import signal

from app.config.settings import settings
from app.db.chunks_repository import create_chunks_repository
from app.ingestion.embedder import GeminiEmbedder, MinIntervalRateLimiter
from worker.processing import process_document

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def run() -> None:
    missing = settings.missing_worker_settings()
    if missing:
        raise RuntimeError(f"Worker configuration is incomplete: {', '.join(missing)}")

    repository = create_chunks_repository(
        settings.supabase_url or "", settings.supabase_service_role_key or ""
    )

    rate_limiter = MinIntervalRateLimiter(settings.embed_min_seconds_between_requests)
    embedder = GeminiEmbedder(
        api_key=settings.google_api_key or "",
        model=settings.gemini_embed_model,
        output_dim=settings.embed_output_dim,
        rate_limiter=rate_limiter,
        max_retries=settings.embed_max_retries,
        retry_base_delay=settings.embed_retry_base_delay,
    )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def request_stop() -> None:
        logger.info("SIGTERM received; completing the current batch before exit")
        stop_event.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, request_stop)
        except NotImplementedError:  # Windows development environments
            signal.signal(signum, lambda *_: request_stop())

    semaphore = asyncio.Semaphore(settings.worker_max_concurrency)

    async def process_claim(claim: dict[str, str]) -> None:
        async with semaphore:
            await asyncio.to_thread(
                process_document,
                claim["id"],
                claim["storage_key"],
                settings=settings,
                repository=repository,
                embedder=embedder,
                rate_limiter=rate_limiter,
            )

    while not stop_event.is_set():
        repository.sweep_stale_processing(settings.worker_stale_processing_minutes)
        claims = repository.claim_pending(settings.worker_batch_size)
        if claims:
            results = await asyncio.gather(
                *(process_claim(claim) for claim in claims), return_exceptions=True
            )
            for result in results:
                if isinstance(result, Exception):
                    logger.exception("Unhandled document-processing error", exc_info=result)
            continue
        try:
            await asyncio.wait_for(
                stop_event.wait(), timeout=settings.worker_poll_interval_seconds
            )
        except TimeoutError:
            pass

    logger.info("Worker stopped cleanly")


if __name__ == "__main__":
    asyncio.run(run())
