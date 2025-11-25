"""DTOs for issue list responses."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class IssueListItemDTO:
    """Single issue item in a list."""

    id: str
    external_id: str
    title: Optional[str]
    status: str
    priority: Optional[int]
    created_at: str
    source_name: Optional[str] = None
    source_type: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "external_id": self.external_id,
            "title": self.title,
            "status": self.status,
            "priority": self.priority,
            "created_at": self.created_at,
            "source_name": self.source_name,
            "source_type": self.source_type,
        }


@dataclass
class IssueListResponseDTO:
    """Paginated list of issues."""

    items: List[IssueListItemDTO]
    total: int
    limit: int
    offset: int

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "items": [item.to_dict() for item in self.items],
            "total": self.total,
            "limit": self.limit,
            "offset": self.offset,
        }
