"""Source repository interface."""

from abc import ABC, abstractmethod
from uuid import UUID

from ..models import Source, SourceType


class SourceRepository(ABC):
    """Abstract repository for Source entities."""

    @abstractmethod
    async def get_by_id(self, source_id: UUID) -> Source | None:
        """Get source by ID."""
        pass

    @abstractmethod
    async def get_by_type(self, source_type: SourceType) -> list[Source]:
        """Get all sources of a specific type."""
        pass

    @abstractmethod
    async def create(self, source: Source) -> Source:
        """Create a new source."""
        pass
