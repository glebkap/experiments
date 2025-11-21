"""Configuration settings for Parser Service."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database Configuration
    db_host: str = "localhost"
    db_port: int = 15432
    db_name: str = "support"
    db_user: str = "support"
    db_password: str = ""

    # Analyzer Service
    analyzer_url: str = "http://analyzer:8002"

    # Parser Service
    parser_host: str = "0.0.0.0"
    parser_port: int = 8001

    # File Upload
    upload_dir: str = "/tmp/uploads"
    max_upload_size: int = 104857600  # 100MB

    # Logging
    log_level: str = "INFO"

    @property
    def database_url(self) -> str:
        """Construct PostgreSQL connection URL."""
        password_part = f":{self.db_password}" if self.db_password else ""
        return (
            f"postgresql://{self.db_user}{password_part}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


# Global settings instance
settings = Settings()
