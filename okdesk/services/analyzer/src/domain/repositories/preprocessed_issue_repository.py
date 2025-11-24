"""Repository interface for PreprocessedIssue."""

from abc import ABC, abstractmethod
from typing import List
from uuid import UUID

from ..models.preprocessed_issue import PreprocessedIssue


class PreprocessedIssueRepository(ABC):
    """Abstract repository for PreprocessedIssue operations."""

    @abstractmethod
    async def save_batch(self, items: List[PreprocessedIssue]) -> None:
        """
        Save batch of preprocessed issues.

        Uses batch insert for performance.

        Args:
            items: List of preprocessed issues to save
        """
        pass

    @abstractmethod
    async def save(self, item: PreprocessedIssue) -> None:
        """
        Save single preprocessed issue.

        Args:
            item: Preprocessed issue to save
        """
        pass

    @abstractmethod
    async def exists(self, issue_id: UUID) -> bool:
        """
        Check if issue has been preprocessed.

        Args:
            issue_id: Issue UUID to check

        Returns:
            True if issue exists in preprocessed_issues, False otherwise
        """
        pass

    @abstractmethod
    async def get_by_id(self, issue_id: UUID) -> PreprocessedIssue | None:
        """
        Get preprocessed issue by ID.

        Args:
            issue_id: Issue UUID

        Returns:
            PreprocessedIssue if found, None otherwise
        """
        pass

    @abstractmethod
    async def delete(self, issue_id: UUID) -> None:
        """
        Delete preprocessed issue (for reprocessing).

        Args:
            issue_id: Issue UUID to delete
        """
        pass

    @abstractmethod
    async def count_total(self) -> int:
        """
        Count total number of preprocessed issues.

        Returns:
            Number of preprocessed issues
        """
        pass

    @abstractmethod
    async def clear_all(self) -> int:
        """
        Delete all preprocessed issues (for full reprocessing).

        WARNING: This will mark all issues as unprocessed!

        Returns:
            Number of deleted records
        """
        pass
