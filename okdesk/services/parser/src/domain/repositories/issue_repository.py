"""Issue repository interface."""

from abc import ABC, abstractmethod
from uuid import UUID

from ..models import Issue


class IssueRepository(ABC):
    """Abstract repository for Issue entities."""

    @abstractmethod
    async def get_by_id(self, issue_id: UUID) -> Issue | None:
        """Get issue by ID."""
        pass

    @abstractmethod
    async def get_by_external_id(self, external_id: str, source_id: UUID) -> Issue | None:
        """Get issue by external_id and source_id."""
        pass

    @abstractmethod
    async def create(self, issue: Issue) -> Issue:
        """Create a new issue."""
        pass

    @abstractmethod
    async def update(self, issue: Issue) -> Issue:
        """Update an existing issue."""
        pass
