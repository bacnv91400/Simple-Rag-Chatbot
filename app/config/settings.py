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


settings = Settings()
