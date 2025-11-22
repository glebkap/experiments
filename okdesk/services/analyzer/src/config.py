"""Application configuration using pydantic-settings."""

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
    embedding_dim: int = 1024
    max_seq_length: int = 512
    device: str = "cpu"  # cpu or cuda

    # Pipeline
    batch_size: int = 100
    embedding_batch_size: int = 32
    max_retries: int = 3
    retry_delay: int = 5

    # Processing Manager
    auto_start_processing: bool = False  # Auto-start background processing on startup
    poll_interval_seconds: float = 1.0  # Interval to poll for new issues (seconds)

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


# Global settings instance
settings = Settings()
