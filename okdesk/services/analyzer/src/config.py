"""Application configuration using pydantic-settings."""

from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = "postgresql+asyncpg://support:support@localhost:15432/support"
    db_pool_size: int = 10
    db_echo: bool = False

    # ChromaDB
    chromadb_host: str = "localhost"
    chromadb_port: int = 8100
    chromadb_collection: str = "support_issues"

    # ML Models
    embedding_model: str = "intfloat/multilingual-e5-large"
    max_seq_length: int = 512
    device: Literal["cpu", "cuda", "mps", "auto"] = "cpu"  # cpu, cuda (NVIDIA GPU), mps (Apple Silicon), auto

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """
        Validate device setting.

        Args:
            v: Device value

        Returns:
            Validated device string

        Raises:
            ValueError: If device is not supported
        """
        allowed = ["cpu", "cuda", "mps", "auto"]
        if v not in allowed:
            raise ValueError(f"device must be one of {allowed}, got '{v}'")
        return v

    # Pipeline
    batch_size: int = 100
    embedding_batch_size: int = 32
    max_retries: int = 3
    retry_delay: int = 5

    # Processing Manager
    auto_start_processing: bool = False  # Auto-start background processing on startup
    poll_interval_seconds: float = 1.0  # Interval to poll for new issues (seconds)
    reprocess_all: bool = False  # Force reprocessing of all issues (ignores preprocessed status)

    # Clustering
    clustering_method: str = "hdbscan"  # hdbscan or kmeans
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int = 3
    kmeans_max_k: int = 20

    # API
    api_title: str = "Analyzer Service"
    api_version: str = "1.0.0"
    api_description: str = "Pipeline processing service for issue analysis"
    api_host: str = "0.0.0.0"
    api_port: int = 8002

    # Logging
    log_level: str = "INFO"
    debug_log_issue_content: bool = True  # Log full issue text in DEBUG mode
    debug_log_max_content_length: int = 500  # Max length of content to log (0 = unlimited)
    debug_log_sample_size: int = 3  # Number of sample issues to log in detail


# Global settings instance
settings = Settings()
