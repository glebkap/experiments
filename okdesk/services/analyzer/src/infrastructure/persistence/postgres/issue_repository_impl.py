"""PostgreSQL implementation of IssueRepository."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import and_, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ....domain.models.issue import Issue
from ....domain.models.message import Message
from ....domain.repositories.issue_repository import IssueRepository
from .models import IssueModel, MessageModel, PreprocessedIssueModel, SourceModel

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

    async def get_unprocessed_issues(self, limit: int) -> List[Issue]:
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

    async def get_by_ids(self, issue_ids: List[UUID]) -> List[Issue]:
        """
        Get issues by IDs.

        Args:
            issue_ids: List of issue UUIDs

        Returns:
            List of Issue domain objects
        """
        if not issue_ids:
            return []

        query = select(IssueModel).where(IssueModel.id.in_(issue_ids))
        result = await self.session.execute(query)
        models = result.scalars().all()

        logger.debug(f"Found {len(models)} issues out of {len(issue_ids)} requested")

        return [self._to_domain(model) for model in models]

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

    def _message_to_domain(self, model: MessageModel) -> Message:
        """Convert MessageModel to domain Message."""
        return Message(
            id=model.id,
            external_id=model.external_id,
            issue_id=model.issue_id,
            author_id=model.author_id,
            author_name=model.author_name,
            author_type=model.author_type,
            content=model.content,
            is_public=model.is_public,
            published_at=model.published_at,
        )

    # ==================== Extension methods for 04-analyzer-extensions ====================

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
        """Get issues with filtering and pagination."""
        query = select(IssueModel)

        # Build filter conditions
        conditions = []
        if status:
            conditions.append(IssueModel.status == status)
        if source_id:
            conditions.append(IssueModel.source_id == source_id)
        if priority:
            conditions.append(IssueModel.priority == priority)
        if date_from:
            conditions.append(IssueModel.created_at >= date_from)
        if date_to:
            conditions.append(IssueModel.created_at <= date_to)

        if conditions:
            query = query.where(and_(*conditions))

        query = query.order_by(IssueModel.created_at.desc()).limit(limit).offset(offset)

        result = await self.session.execute(query)
        models = result.scalars().all()

        logger.debug(f"Found {len(models)} issues with filters")
        return [self._to_domain(model) for model in models]

    async def get_issue_with_messages(
        self, issue_id: UUID
    ) -> Optional[Tuple[Issue, List[Message]]]:
        """Get issue with all its messages."""
        query = (
            select(IssueModel)
            .options(selectinload(IssueModel.messages))
            .where(IssueModel.id == issue_id)
        )

        result = await self.session.execute(query)
        model = result.scalar_one_or_none()

        if model is None:
            logger.debug(f"Issue {issue_id} not found")
            return None

        issue = self._to_domain(model)
        messages = [self._message_to_domain(m) for m in model.messages]

        # Sort messages by published_at
        messages.sort(key=lambda m: m.published_at or datetime.min)

        logger.debug(f"Found issue {issue_id} with {len(messages)} messages")
        return (issue, messages)

    async def count_issues_with_filters(
        self,
        status: Optional[str] = None,
        source_id: Optional[UUID] = None,
        priority: Optional[int] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> int:
        """Count issues with filters."""
        query = select(func.count()).select_from(IssueModel)

        conditions = []
        if status:
            conditions.append(IssueModel.status == status)
        if source_id:
            conditions.append(IssueModel.source_id == source_id)
        if priority:
            conditions.append(IssueModel.priority == priority)
        if date_from:
            conditions.append(IssueModel.created_at >= date_from)
        if date_to:
            conditions.append(IssueModel.created_at <= date_to)

        if conditions:
            query = query.where(and_(*conditions))

        result = await self.session.execute(query)
        count = result.scalar_one()

        logger.debug(f"Count with filters: {count}")
        return count

    async def fulltext_search(
        self,
        query: str,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[Issue], int]:
        """Full-text search using PostgreSQL FTS on preprocessed_issues."""
        if not query.strip():
            return [], 0

        # Use PostgreSQL full-text search with Russian language config
        # First try plainto_tsquery, then fall back to websearch_tsquery for complex queries
        search_query = select(
            IssueModel,
            func.ts_rank(
                func.to_tsvector("russian", PreprocessedIssueModel.content),
                func.plainto_tsquery("russian", query),
            ).label("rank"),
        ).join(PreprocessedIssueModel, IssueModel.id == PreprocessedIssueModel.id)

        search_query = search_query.where(
            func.to_tsvector("russian", PreprocessedIssueModel.content).match(
                func.plainto_tsquery("russian", query)
            )
        )

        # Count total matching
        count_query = (
            select(func.count())
            .select_from(IssueModel)
            .join(PreprocessedIssueModel, IssueModel.id == PreprocessedIssueModel.id)
            .where(
                func.to_tsvector("russian", PreprocessedIssueModel.content).match(
                    func.plainto_tsquery("russian", query)
                )
            )
        )

        count_result = await self.session.execute(count_query)
        total_count = count_result.scalar_one()

        # Get paginated results ordered by relevance
        search_query = (
            search_query.order_by(text("rank DESC")).limit(limit).offset(offset)
        )

        result = await self.session.execute(search_query)
        rows = result.all()

        issues = [self._to_domain(row[0]) for row in rows]

        logger.info(f"Full-text search '{query}': found {total_count} results")
        return issues, total_count

    async def get_issue_source_info(self, issue_id: UUID) -> Optional[Dict[str, Any]]:
        """Get issue with source information."""
        query = (
            select(IssueModel, SourceModel)
            .join(SourceModel, IssueModel.source_id == SourceModel.id)
            .where(IssueModel.id == issue_id)
        )

        result = await self.session.execute(query)
        row = result.one_or_none()

        if row is None:
            return None

        issue_model, source_model = row
        return {
            "issue": self._to_domain(issue_model),
            "source_name": source_model.name,
            "source_type": source_model.type,
        }
