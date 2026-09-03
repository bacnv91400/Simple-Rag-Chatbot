"""Small environment-based configuration for local development."""

from dataclasses import dataclass
from os import getenv


@dataclass(frozen=True)
class Settings:
    """Configuration values"""

    app_name: str = getenv("APP_NAME")
    app_env: str = getenv("APP_ENV")


settings = Settings()
