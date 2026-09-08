from app.config.settings import Settings


def test_default_settings_are_available() -> None:
    default_settings = Settings()

    assert default_settings.app_name
    assert default_settings.app_env == "development"
    assert default_settings.embedding_dim == 1024
    assert default_settings.worker_max_concurrency == 3
