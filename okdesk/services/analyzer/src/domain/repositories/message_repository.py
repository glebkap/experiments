"""Repository interface for Message entity."""

from abc import ABC, abstractmethod
from typing import List
from uuid import UUID

from ..models.message import Message


class MessageRepository(ABC):
    """Abstract repository for Message operations."""

    @abstractmethod
    async def get_by_issue_id(self, issue_id: UUID) -> List[Message]:
        """
        Get all messages for an issue.

        Args:
            issue_id: Issue UUID

        Returns:
            List of messages for the issue, ordered by published_at
        """
        pass

    @abstractmethod
    async def get_by_ids(self, message_ids: List[UUID]) -> List[Message]:
        """
        Get multiple messages by IDs.

        Args:
            message_ids: List of message UUIDs

        Returns:
            List of found messages
        """
        pass

    @abstractmethod
    async def count_by_issue_id(self, issue_id: UUID) -> int:
        """
        Count messages for an issue.

        Args:
            issue_id: Issue UUID

        Returns:
            Number of messages
        """
        pass
