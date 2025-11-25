"""Configuration management for support-cli."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API URLs
    api_gateway_url: HttpUrl = Field(
        default="http://localhost:8000",
        description="API Gateway URL",
    )
    parser_service_url: HttpUrl = Field(
        default="http://localhost:8001",
        description="Parser Service URL (fallback)",
    )
    analyzer_service_url: HttpUrl = Field(
        default="http://localhost:8002",
        description="Analyzer Service URL (fallback)",
    )

    # Processing defaults
    default_batch_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Default batch size for processing",
    )
    default_top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Default number of results for search",
    )
    default_min_similarity: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Default minimum similarity threshold",
    )

    # HTTP client settings
    request_timeout: int = Field(
        default=300,
        ge=1,
        description="Request timeout in seconds",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retries for failed requests",
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    log_format: Literal["simple", "detailed", "json"] = Field(
        default="simple",
        description="Log format style",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings: Cached settings instance
    """
    return Settings()
