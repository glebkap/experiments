"""PostgreSQL implementation of SourceRepository."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ....domain.models import Source, SourceType
from ....domain.repositories import SourceRepository
from ..mappers import SourceMapper
from ..models import SourceModel


class SourceRepositoryImpl(SourceRepository):
    """PostgreSQL implementation of SourceRepository."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session."""
        self._session = session
        self._mapper = SourceMapper()

    async def get_by_id(self, source_id: UUID) -> Source | None:
        """Get source by ID."""
        result = await self._session.execute(
            select(SourceModel).where(SourceModel.id == source_id)
        )
        model = result.scalar_one_or_none()
        return self._mapper.to_domain(model) if model else None

    async def get_by_type(self, source_type: SourceType) -> list[Source]:
        """Get all sources of a specific type."""
        result = await self._session.execute(
            select(SourceModel).where(SourceModel.type == source_type)
        )
        models = result.scalars().all()
        return [self._mapper.to_domain(model) for model in models]

    async def create(self, source: Source) -> Source:
        """Create a new source."""
        model = self._mapper.to_model(source)
        self._session.add(model)
        await self._session.flush()
        await self._session.refresh(model)
        return self._mapper.to_domain(model)
