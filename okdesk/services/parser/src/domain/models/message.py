"""Message domain model and types."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from uuid import UUID


class AuthorType(str, Enum):
    """Type of message author."""

    EMPLOYEE = "employee"
    CONTACT = "contact"
    USER = "user"


@dataclass
class Message:
    """Message entity within an issue."""

    id: UUID
    issue_id: UUID
    external_id: str
    author_id: str | None
    author_name: str | None
    author_type: AuthorType | None
    content: str
    is_public: bool
    published_at: datetime | None

    def __post_init__(self) -> None:
        """Validate entity after initialization."""
        if not self.external_id:
            raise ValueError("Message external_id cannot be empty")
        if not self.content:
            raise ValueError("Message content cannot be empty")
