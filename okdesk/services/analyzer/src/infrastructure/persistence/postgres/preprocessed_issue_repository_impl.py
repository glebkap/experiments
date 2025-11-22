"""PostgreSQL implementation of PreprocessedIssueRepository."""

import logging
from typing import List
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from ....domain.models.preprocessed_issue import PreprocessedIssue
from ....domain.repositories.preprocessed_issue_repository import (
    PreprocessedIssueRepository,
)
from .models import PreprocessedIssueModel

logger = logging.getLogger(__name__)


class PreprocessedIssueRepositoryImpl(PreprocessedIssueRepository):
    """PostgreSQL implementation of PreprocessedIssueRepository."""

    def __init__(self, session: AsyncSession):
        """
        Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        self.session = session

    async def save_batch(self, items: List[PreprocessedIssue]) -> None:
        """
        Save multiple preprocessed issues in a batch.

        Uses UPSERT (INSERT ... ON CONFLICT) for efficiency.

        Args:
            items: List of PreprocessedIssue domain objects
        """
        if not items:
            logger.debug("No items to save")
            return

        logger.debug(f"Saving batch of {len(items)} preprocessed issues")

        # Prepare data for bulk insert
        values = [
            {
                "id": item.issue_id,
                "content": item.content,
            }
            for item in items
        ]

        # Use PostgreSQL INSERT ... ON CONFLICT DO UPDATE
        stmt = insert(PreprocessedIssueModel).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["id"],
            set_={"content": stmt.excluded.content, "processed_at": stmt.excluded.processed_at},
        )

        await self.session.execute(stmt)
        await self.session.commit()

        logger.info(f"Successfully saved {len(items)} preprocessed issues")

    async def exists(self, issue_id: UUID) -> bool:
        """
        Check if issue has been preprocessed.

        Args:
            issue_id: Issue UUID

        Returns:
            True if preprocessed record exists
        """
        query = select(PreprocessedIssueModel.id).where(
            PreprocessedIssueModel.id == issue_id
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none() is not None

    async def delete(self, issue_id: UUID) -> None:
        """
        Delete preprocessed issue by ID.

        Args:
            issue_id: Issue UUID
        """
        stmt = delete(PreprocessedIssueModel).where(
            PreprocessedIssueModel.id == issue_id
        )
        result = await self.session.execute(stmt)
        await self.session.commit()

        if result.rowcount > 0:
            logger.info(f"Deleted preprocessed issue {issue_id}")
        else:
            logger.debug(f"No preprocessed issue found with ID {issue_id}")

    async def get_by_id(self, issue_id: UUID) -> PreprocessedIssue | None:
        """
        Get preprocessed issue by ID.

        Args:
            issue_id: Issue UUID

        Returns:
            PreprocessedIssue or None if not found
        """
        query = select(PreprocessedIssueModel).where(
            PreprocessedIssueModel.id == issue_id
        )
        result = await self.session.execute(query)
        model = result.scalar_one_or_none()

        if model is None:
            logger.debug(f"Preprocessed issue {issue_id} not found")
            return None

        return self._to_domain(model)

    def _to_domain(self, model: PreprocessedIssueModel) -> PreprocessedIssue:
        """
        Convert SQLAlchemy model to domain object.

        Args:
            model: PreprocessedIssueModel instance

        Returns:
            PreprocessedIssue domain object
        """
        return PreprocessedIssue(
            issue_id=model.id,
            content=model.content,
            processed_at=model.processed_at,
        )
