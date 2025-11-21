"""Issue domain model and types."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from uuid import UUID


class IssueStatus(str, Enum):
    """Status of an issue."""

    OPENED = "opened"
    WAIT = "wait"
    COMPLETED = "completed"
    CLOSED = "closed"


@dataclass
class Issue:
    """Support issue entity."""

    id: UUID
    external_id: str
    source_id: UUID
    title: str | None
    description: str | None
    status: IssueStatus | None
    priority: int | None
    created_at: datetime | None
    updated_at: datetime | None
    completed_at: datetime | None

    def __post_init__(self) -> None:
        """Validate entity after initialization."""
        if not self.external_id:
            raise ValueError("Issue external_id cannot be empty")
        if self.priority is not None and not (1 <= self.priority <= 4):
            raise ValueError(f"Issue priority must be between 1 and 4, got {self.priority}")
