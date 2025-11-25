"""DTOs for export functionality."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any


@dataclass
class ExportFiltersDTO:
    """Filters for export."""

    status: Optional[str] = None
    source_id: Optional[str] = None
    priority: Optional[int] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None


@dataclass
class ExportRequestDTO:
    """Request for data export."""

    format: str  # "csv" or "json"
    filters: Optional[ExportFiltersDTO] = None


@dataclass
class ExportResultDTO:
    """Result of export operation."""

    filename: str
    format: str
    total_records: int
    content_type: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "filename": self.filename,
            "format": self.format,
            "total_records": self.total_records,
            "content_type": self.content_type,
        }
