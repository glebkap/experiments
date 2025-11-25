"""DTO for Issue entity."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class IssueDTO:
    """Data transfer object for Issue entity."""

    id: str
    external_id: str
    source_id: str
    title: Optional[str]
    description: Optional[str]
    status: str
    priority: Optional[int]
    created_at: str
    updated_at: Optional[str]

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "id": self.id,
            "external_id": self.external_id,
            "source_id": self.source_id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
