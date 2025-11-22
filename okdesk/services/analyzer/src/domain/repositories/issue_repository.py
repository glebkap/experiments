"""Repository interface for Issue entity."""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from ..models.issue import Issue


class IssueRepository(ABC):
    """Abstract repository for Issue operations."""

    @abstractmethod
    async def get_unprocessed_issues(self, limit: int) -> List[Issue]:
        """
        Get issues that have not been preprocessed yet.

        Issues are considered unprocessed if they don't exist in preprocessed_issues table.

        Args:
            limit: Maximum number of issues to return

        Returns:
            List of unprocessed issues
        """
        pass

    @abstractmethod
    async def get_by_id(self, issue_id: UUID) -> Optional[Issue]:
        """
        Get issue by ID.

        Args:
            issue_id: Issue UUID

        Returns:
            Issue if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_by_ids(self, issue_ids: List[UUID]) -> List[Issue]:
        """
        Get multiple issues by IDs.

        Args:
            issue_ids: List of issue UUIDs

        Returns:
            List of found issues
        """
        pass

    @abstractmethod
    async def count_unprocessed(self) -> int:
        """
        Count total number of unprocessed issues.

        Returns:
            Number of unprocessed issues
        """
        pass

    @abstractmethod
    async def count_total(self) -> int:
        """
        Count total number of issues.

        Returns:
            Total number of issues
        """
        pass
