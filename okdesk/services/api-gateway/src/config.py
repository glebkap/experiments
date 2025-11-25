"""Configuration for API Gateway Service."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """API Gateway configuration settings."""

    # Service URLs
    parser_url: str = "http://parser:8001"
    analyzer_url: str = "http://analyzer:8002"

    # API Gateway settings
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS settings
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Timeouts (in seconds)
    request_timeout: int = 30
    health_check_timeout: int = 5

    # Logging
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# Global settings instance
settings = Settings()
