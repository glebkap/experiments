"""Import repository interface."""

from abc import ABC, abstractmethod
from uuid import UUID

from ..models import ImportJob


class ImportRepository(ABC):
    """Abstract repository for ImportJob entities."""

    @abstractmethod
    async def get_by_id(self, import_id: UUID) -> ImportJob | None:
        """Get import job by ID."""
        pass

    @abstractmethod
    async def create(self, import_job: ImportJob) -> ImportJob:
        """Create a new import job."""
        pass

    @abstractmethod
    async def update(self, import_job: ImportJob) -> ImportJob:
        """Update an existing import job."""
        pass

    @abstractmethod
    async def list_all(self, limit: int = 100, offset: int = 0) -> list[ImportJob]:
        """List all import jobs with pagination."""
        pass
