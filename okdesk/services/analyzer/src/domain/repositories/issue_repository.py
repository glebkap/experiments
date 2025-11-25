"""Repository interface for Issue entity."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID

from ..models.issue import Issue
from ..models.message import Message


class IssueRepository(ABC):
    """Abstract repository for Issue operations."""

    @abstractmethod
    async def get_unprocessed_issues(self, limit: int, reprocess_all: bool = False) -> List[Issue]:
        """
        Get issues that have not been preprocessed yet.

        Issues are considered unprocessed if they don't exist in preprocessed_issues table.
        If reprocess_all is True, returns all issues regardless of processing status.

        Args:
            limit: Maximum number of issues to return
            reprocess_all: If True, ignore preprocessing status and return all issues

        Returns:
            List of unprocessed (or all) issues
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

    # ==================== Extension methods for 04-analyzer-extensions ====================

    @abstractmethod
    async def get_issues_with_filters(
        self,
        status: Optional[str] = None,
        source_id: Optional[UUID] = None,
        priority: Optional[int] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Issue]:
        """
        Get issues with filtering and pagination.

        Args:
            status: Filter by issue status (opened, wait, completed, closed)
            source_id: Filter by source UUID
            priority: Filter by priority (1-4)
            date_from: Filter by created_at >= date_from
            date_to: Filter by created_at <= date_to
            limit: Maximum number of issues to return
            offset: Number of issues to skip

        Returns:
            List of filtered issues
        """
        pass

    @abstractmethod
    async def get_issue_with_messages(
        self, issue_id: UUID
    ) -> Optional[Tuple[Issue, List[Message]]]:
        """
        Get issue with all its messages.

        Args:
            issue_id: Issue UUID

        Returns:
            Tuple of (Issue, List[Message]) if found, None otherwise
        """
        pass

    @abstractmethod
    async def count_issues_with_filters(
        self,
        status: Optional[str] = None,
        source_id: Optional[UUID] = None,
        priority: Optional[int] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> int:
        """
        Count issues with filters.

        Args:
            status: Filter by issue status
            source_id: Filter by source UUID
            priority: Filter by priority
            date_from: Filter by created_at >= date_from
            date_to: Filter by created_at <= date_to

        Returns:
            Number of matching issues
        """
        pass

    @abstractmethod
    async def fulltext_search(
        self,
        query: str,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[Issue], int]:
        """
        Full-text search using PostgreSQL FTS on preprocessed_issues.

        Args:
            query: Search query text
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            Tuple of (List of matching issues, total count)
        """
        pass

    @abstractmethod
    async def get_issue_source_info(self, issue_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get issue with source information.

        Args:
            issue_id: Issue UUID

        Returns:
            Dict with issue and source info, or None if not found
        """
        pass
