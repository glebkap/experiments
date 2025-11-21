"""Data Transfer Objects for import operations."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID


@dataclass
class ImportRequest:
    """Request to import data from file."""

    file_path: str
    source_id: UUID
    filename: str | None = None


@dataclass
class ImportStats:
    """Statistics about import operation."""

    total_issues: int
    new_issues: int
    updated_issues: int
    total_messages: int
    new_messages: int
    updated_messages: int


@dataclass
class ImportProgress:
    """Progress information for ongoing import."""

    import_id: UUID
    status: str  # in_progress, completed, failed
    started_at: datetime
    completed_at: datetime | None
    stats: ImportStats | None
    error_message: str | None


@dataclass
class ImportResponse:
    """Response after starting import."""

    import_id: UUID
    status: str
    message: str
