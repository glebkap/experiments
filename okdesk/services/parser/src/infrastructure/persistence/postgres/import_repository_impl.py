"""PostgreSQL implementation of ImportRepository."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ....domain.models import ImportJob
from ....domain.repositories import ImportRepository
from ..mappers import ImportJobMapper
from ..models import ImportModel


class ImportRepositoryImpl(ImportRepository):
    """PostgreSQL implementation of ImportRepository."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session."""
        self._session = session
        self._mapper = ImportJobMapper()

    async def get_by_id(self, import_id: UUID) -> ImportJob | None:
        """Get import job by ID."""
        result = await self._session.execute(
            select(ImportModel).where(ImportModel.id == import_id)
        )
        model = result.scalar_one_or_none()
        return self._mapper.to_domain(model) if model else None

    async def create(self, import_job: ImportJob) -> ImportJob:
        """Create a new import job."""
        model = self._mapper.to_model(import_job)
        self._session.add(model)
        await self._session.flush()
        await self._session.refresh(model)
        return self._mapper.to_domain(model)

    async def update(self, import_job: ImportJob) -> ImportJob:
        """Update an existing import job."""
        result = await self._session.execute(
            select(ImportModel).where(ImportModel.id == import_job.id)
        )
        model = result.scalar_one_or_none()
        if not model:
            raise ValueError(f"ImportJob with id {import_job.id} not found")

        # Update fields
        model.source_id = import_job.source_id
        model.filename = import_job.filename
        model.file_path = import_job.file_path
        model.started_at = import_job.started_at
        model.completed_at = import_job.completed_at
        model.status = import_job.status
        model.stats = import_job.stats
        model.error_message = import_job.error_message

        await self._session.flush()
        await self._session.refresh(model)
        return self._mapper.to_domain(model)

    async def list_all(self, limit: int = 100, offset: int = 0) -> list[ImportJob]:
        """List all import jobs with pagination."""
        result = await self._session.execute(
            select(ImportModel).order_by(ImportModel.started_at.desc()).limit(limit).offset(offset)
        )
        models = result.scalars().all()
        return [self._mapper.to_domain(model) for model in models]
