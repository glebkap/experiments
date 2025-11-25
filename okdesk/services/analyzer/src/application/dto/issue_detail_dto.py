"""DTOs for issue detail responses."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class MessageDTO:
    """Message in an issue."""

    id: str
    external_id: str
    author_name: Optional[str]
    author_type: str
    content: str
    is_public: bool
    published_at: Optional[str]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "external_id": self.external_id,
            "author_name": self.author_name,
            "author_type": self.author_type,
            "content": self.content,
            "is_public": self.is_public,
            "published_at": self.published_at,
        }


@dataclass
class IssueDetailDTO:
    """Detailed issue information with messages."""

    id: str
    external_id: str
    title: Optional[str]
    description: Optional[str]
    status: str
    priority: Optional[int]
    created_at: str
    updated_at: Optional[str]
    completed_at: Optional[str]
    source_name: Optional[str]
    source_type: Optional[str]
    messages: List[MessageDTO] = field(default_factory=list)
    cluster_id: Optional[str] = None
    cluster_label: Optional[int] = None
    distance_to_centroid: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "external_id": self.external_id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "source_name": self.source_name,
            "source_type": self.source_type,
            "messages": [m.to_dict() for m in self.messages],
            "cluster_id": self.cluster_id,
            "cluster_label": self.cluster_label,
            "distance_to_centroid": self.distance_to_centroid,
        }
