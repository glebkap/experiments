"""ImportJob domain model and types."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID


class ImportStatus(str, Enum):
    """Status of an import job."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ImportJob:
    """Import job entity tracking data import process."""

    id: UUID
    source_id: UUID
    filename: str | None
    file_path: str | None
    started_at: datetime
    completed_at: datetime | None
    status: ImportStatus
    stats: dict[str, Any] | None
    error_message: str | None

    def __post_init__(self) -> None:
        """Validate entity after initialization."""
        if not isinstance(self.status, ImportStatus):
            raise ValueError(f"Invalid import status: {self.status}")

    def mark_completed(self, stats: dict[str, Any]) -> None:
        """Mark import as completed with statistics."""
        self.status = ImportStatus.COMPLETED
        self.completed_at = datetime.now()
        self.stats = stats
        self.error_message = None

    def mark_failed(self, error: str) -> None:
        """Mark import as failed with error message."""
        self.status = ImportStatus.FAILED
        self.completed_at = datetime.now()
        self.error_message = error
