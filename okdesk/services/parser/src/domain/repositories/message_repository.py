"""Message repository interface."""

from abc import ABC, abstractmethod
from uuid import UUID

from ..models import Message


class MessageRepository(ABC):
    """Abstract repository for Message entities."""

    @abstractmethod
    async def get_by_id(self, message_id: UUID) -> Message | None:
        """Get message by ID."""
        pass

    @abstractmethod
    async def get_by_external_id(self, external_id: str, issue_id: UUID) -> Message | None:
        """Get message by external_id and issue_id."""
        pass

    @abstractmethod
    async def get_by_issue_id(self, issue_id: UUID) -> list[Message]:
        """Get all messages for an issue."""
        pass

    @abstractmethod
    async def create(self, message: Message) -> Message:
        """Create a new message."""
        pass

    @abstractmethod
    async def update(self, message: Message) -> Message:
        """Update an existing message."""
        pass

    @abstractmethod
    async def bulk_create(self, messages: list[Message]) -> list[Message]:
        """Create multiple messages in bulk."""
        pass
