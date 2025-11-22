"""PostgreSQL implementation of IssueRepository."""

import logging
from typing import List, Optional
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ....domain.models.issue import Issue
from ....domain.repositories.issue_repository import IssueRepository
from .models import IssueModel, PreprocessedIssueModel

logger = logging.getLogger(__name__)


class IssueRepositoryImpl(IssueRepository):
    """PostgreSQL implementation of IssueRepository."""

    def __init__(self, session: AsyncSession):
        """
        Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        self.session = session

    async def get_unprocessed(self, limit: int = 100) -> List[Issue]:
        """
        Get unprocessed issues (not in preprocessed_issues table).

        Args:
            limit: Maximum number of issues to return

        Returns:
            List of unprocessed Issue domain objects
        """
        logger.debug(f"Fetching up to {limit} unprocessed issues")

        # Query for issues that don't have a corresponding preprocessed record
        query = (
            select(IssueModel)
            .outerjoin(
                PreprocessedIssueModel, IssueModel.id == PreprocessedIssueModel.id
            )
            .where(PreprocessedIssueModel.id.is_(None))
            .limit(limit)
            .order_by(IssueModel.created_at.desc())
        )

        result = await self.session.execute(query)
        models = result.scalars().all()

        logger.info(f"Found {len(models)} unprocessed issues")

        return [self._to_domain(model) for model in models]

    async def get_by_id(self, issue_id: UUID) -> Optional[Issue]:
        """
        Get issue by ID.

        Args:
            issue_id: Issue UUID

        Returns:
            Issue domain object or None if not found
        """
        query = select(IssueModel).where(IssueModel.id == issue_id)
        result = await self.session.execute(query)
        model = result.scalar_one_or_none()

        if model is None:
            logger.debug(f"Issue {issue_id} not found")
            return None

        return self._to_domain(model)

    async def count_unprocessed(self) -> int:
        """
        Count total number of unprocessed issues.

        Returns:
            Count of unprocessed issues
        """
        # Subquery to get processed issue IDs
        processed_subquery = select(PreprocessedIssueModel.id)

        # Count issues not in the processed set
        query = select(func.count()).select_from(IssueModel).where(
            IssueModel.id.notin_(processed_subquery)
        )

        result = await self.session.execute(query)
        count = result.scalar_one()

        logger.debug(f"Total unprocessed issues: {count}")
        return count

    async def count_total(self) -> int:
        """
        Count total number of issues.

        Returns:
            Total count of all issues
        """
        query = select(func.count()).select_from(IssueModel)
        result = await self.session.execute(query)
        count = result.scalar_one()

        logger.debug(f"Total issues in database: {count}")
        return count

    def _to_domain(self, model: IssueModel) -> Issue:
        """
        Convert SQLAlchemy model to domain object.

        Args:
            model: IssueModel instance

        Returns:
            Issue domain object
        """
        return Issue(
            id=model.id,
            external_id=model.external_id,
            source_id=model.source_id,
            title=model.title,
            description=model.description,
            status=model.status,
            priority=model.priority,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
