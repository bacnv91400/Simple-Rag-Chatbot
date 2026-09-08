"""Small environment-based configuration for local development."""

from dataclasses import dataclass
from os import getenv


@dataclass(frozen=True)
class Settings:
    """Configuration values"""

    app_name: str = getenv("APP_NAME") or "Simple RAG Chatbot"
    app_env: str = getenv("APP_ENV") or "development"
    supabase_url: str | None = getenv("SUPABASE_URL")
    supabase_service_role_key: str | None = getenv("SUPABASE_SERVICE_ROLE_KEY")
    idrive_e2_endpoint: str | None = getenv("IDRIVE_E2_ENDPOINT")
    idrive_e2_region: str | None = getenv("IDRIVE_E2_REGION")
    idrive_e2_access_key: str | None = getenv("IDRIVE_E2_ACCESS_KEY")
    idrive_e2_secret_key: str | None = getenv("IDRIVE_E2_SECRET_KEY")
    idrive_e2_bucket: str | None = getenv("IDRIVE_E2_BUCKET")
    virus_scan_enabled: bool = getenv("VIRUS_SCAN_ENABLED", "true").lower() == "true"
    clamav_host: str = getenv("CLAMAV_HOST", "localhost")
    clamav_port: int = int(getenv("CLAMAV_PORT", "3310"))
    min_image_area_ratio: float = float(getenv("MIN_IMAGE_AREA_RATIO", "0.02"))
    min_image_width: int = int(getenv("MIN_IMAGE_WIDTH", "80"))
    min_image_height: int = int(getenv("MIN_IMAGE_HEIGHT", "80"))
    max_gemini_calls_per_doc: int = int(getenv("MAX_GEMINI_CALLS_PER_DOC", "20"))
    gemini_caption_model: str = getenv("GEMINI_CAPTION_MODEL") or getenv("GEMINI_MODEL") or "gemini-2.0-flash-lite"
    gemini_embed_model: str = getenv("GEMINI_EMBED_MODEL") or getenv("EMBED_MODEL") or "gemini-embedding-001"
    embed_output_dim: int = int(getenv("EMBED_OUTPUT_DIM", "1024"))
    embed_min_seconds_between_requests: float = float(getenv("EMBED_MIN_SECONDS_BETWEEN_REQUESTS", "1.0"))
    embed_max_retries: int = int(getenv("EMBED_MAX_RETRIES", "4"))
    embed_retry_base_delay: float = float(getenv("EMBED_RETRY_BASE_DELAY", "2.0"))
    chunk_tokenizer_model: str | None = getenv("CHUNK_TOKENIZER_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
    chunk_max_tokens: int = int(getenv("CHUNK_MAX_TOKENS", "512"))
    chunk_overlap_tokens: int = int(getenv("CHUNK_OVERLAP_TOKENS", "64"))
    hf_token: str | None = getenv("HF_TOKEN")
    google_api_key: str | None = getenv("GOOGLE_API_KEY")
    embedding_dim: int = int(getenv("EMBEDDING_DIM", "1024"))
    worker_batch_size: int = int(getenv("WORKER_BATCH_SIZE", "5"))
    worker_max_concurrency: int = int(getenv("WORKER_MAX_CONCURRENCY", "3"))
    worker_poll_interval_seconds: float = float(getenv("WORKER_POLL_INTERVAL_SECONDS", "3"))
    worker_stale_processing_minutes: int = int(getenv("WORKER_STALE_PROCESSING_MINUTES", "30"))

    def __post_init__(self) -> None:
        if self.embed_output_dim != self.embedding_dim:
            raise ValueError(
                f"EMBED_OUTPUT_DIM ({self.embed_output_dim}) must equal EMBEDDING_DIM ({self.embedding_dim})"
            )

    def missing_upload_settings(self) -> list[str]:
        """Return required upload configuration keys that are unset."""
        values = {
            "SUPABASE_URL": self.supabase_url,
            "SUPABASE_SERVICE_ROLE_KEY": self.supabase_service_role_key,
            "IDRIVE_E2_ENDPOINT": self.idrive_e2_endpoint,
            "IDRIVE_E2_REGION": self.idrive_e2_region,
            "IDRIVE_E2_ACCESS_KEY": self.idrive_e2_access_key,
            "IDRIVE_E2_SECRET_KEY": self.idrive_e2_secret_key,
            "IDRIVE_E2_BUCKET": self.idrive_e2_bucket,
        }
        return [name for name, value in values.items() if not value]

    def missing_worker_settings(self) -> list[str]:
        """Return configuration required by the background ingestion worker."""
        values = {
            "SUPABASE_URL": self.supabase_url,
            "SUPABASE_SERVICE_ROLE_KEY": self.supabase_service_role_key,
            "IDRIVE_E2_ENDPOINT": self.idrive_e2_endpoint,
            "IDRIVE_E2_REGION": self.idrive_e2_region,
            "IDRIVE_E2_ACCESS_KEY": self.idrive_e2_access_key,
            "IDRIVE_E2_SECRET_KEY": self.idrive_e2_secret_key,
            "IDRIVE_E2_BUCKET": self.idrive_e2_bucket,
            "CHUNK_TOKENIZER_MODEL": self.chunk_tokenizer_model,
            "GOOGLE_API_KEY": self.google_api_key,
            "GEMINI_EMBED_MODEL": self.gemini_embed_model,
            "GEMINI_CAPTION_MODEL": self.gemini_caption_model,
        }
        return [name for name, value in values.items() if not value]


settings = Settings()
