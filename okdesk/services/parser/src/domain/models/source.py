"""Source domain model and types."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID


class SourceType(str, Enum):
    """Type of data source."""

    OKDESK = "okdesk"
    TELEGRAM = "telegram"


@dataclass
class Source:
    """Data source entity."""

    id: UUID
    name: str
    type: SourceType
    config: dict[str, Any] | None
    created_at: datetime

    def __post_init__(self) -> None:
        """Validate entity after initialization."""
        if not self.name:
            raise ValueError("Source name cannot be empty")
        if not isinstance(self.type, SourceType):
            raise ValueError(f"Invalid source type: {self.type}")
