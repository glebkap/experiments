"""PostgreSQL implementation of IssueRepository."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ....domain.models import Issue
from ....domain.repositories import IssueRepository
from ..mappers import IssueMapper
from ..models import IssueModel


class IssueRepositoryImpl(IssueRepository):
    """PostgreSQL implementation of IssueRepository."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session."""
        self._session = session
        self._mapper = IssueMapper()

    async def get_by_id(self, issue_id: UUID) -> Issue | None:
        """Get issue by ID."""
        result = await self._session.execute(
            select(IssueModel).where(IssueModel.id == issue_id)
        )
        model = result.scalar_one_or_none()
        return self._mapper.to_domain(model) if model else None

    async def get_by_external_id(self, external_id: str, source_id: UUID) -> Issue | None:
        """Get issue by external_id and source_id."""
        result = await self._session.execute(
            select(IssueModel).where(
                IssueModel.external_id == external_id,
                IssueModel.source_id == source_id,
            )
        )
        model = result.scalar_one_or_none()
        return self._mapper.to_domain(model) if model else None

    async def create(self, issue: Issue) -> Issue:
        """Create a new issue."""
        model = self._mapper.to_model(issue)
        self._session.add(model)
        await self._session.flush()
        await self._session.refresh(model)
        return self._mapper.to_domain(model)

    async def update(self, issue: Issue) -> Issue:
        """Update an existing issue."""
        result = await self._session.execute(
            select(IssueModel).where(IssueModel.id == issue.id)
        )
        model = result.scalar_one_or_none()
        if not model:
            raise ValueError(f"Issue with id {issue.id} not found")

        # Update fields
        model.external_id = issue.external_id
        model.source_id = issue.source_id
        model.title = issue.title
        model.description = issue.description
        model.status = issue.status
        model.priority = issue.priority
        model.created_at = issue.created_at
        model.updated_at = issue.updated_at
        model.completed_at = issue.completed_at

        await self._session.flush()
        await self._session.refresh(model)
        return self._mapper.to_domain(model)
