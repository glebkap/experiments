"""Pipeline configuration."""

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""

    batch_size: int = 100
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_batch_size: int = 32
    max_retries: int = 3
    retry_delay_seconds: int = 5
    device: str = "cpu"

    def __post_init__(self):
        """Validate configuration."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.embedding_batch_size <= 0:
            raise ValueError(
                f"embedding_batch_size must be positive, got {self.embedding_batch_size}"
            )

        if self.max_retries < 0:
            raise ValueError(f"max_retries cannot be negative, got {self.max_retries}")

        if self.retry_delay_seconds < 0:
            raise ValueError(
                f"retry_delay_seconds cannot be negative, got {self.retry_delay_seconds}"
            )

        if self.device not in ("cpu", "cuda", "mps", "auto"):
            raise ValueError(f"device must be 'cpu', 'cuda', 'mps', or 'auto', got {self.device}")
