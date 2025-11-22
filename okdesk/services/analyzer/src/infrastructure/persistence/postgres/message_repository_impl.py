"""PostgreSQL implementation of MessageRepository."""

import logging
from typing import List
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ....domain.models.message import Message
from ....domain.repositories.message_repository import MessageRepository
from .models import MessageModel

logger = logging.getLogger(__name__)


class MessageRepositoryImpl(MessageRepository):
    """PostgreSQL implementation of MessageRepository."""

    def __init__(self, session: AsyncSession):
        """
        Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        self.session = session

    async def get_by_issue_id(self, issue_id: UUID) -> List[Message]:
        """
        Get all messages for a specific issue.

        Args:
            issue_id: Issue UUID

        Returns:
            List of Message domain objects ordered by creation time
        """
        logger.debug(f"Fetching messages for issue {issue_id}")

        query = (
            select(MessageModel)
            .where(MessageModel.issue_id == issue_id)
            .order_by(MessageModel.created_at.asc())
        )

        result = await self.session.execute(query)
        models = result.scalars().all()

        logger.debug(f"Found {len(models)} messages for issue {issue_id}")

        return [self._to_domain(model) for model in models]

    async def get_by_ids(self, message_ids: List[UUID]) -> List[Message]:
        """
        Get messages by their IDs.

        Args:
            message_ids: List of message UUIDs

        Returns:
            List of Message domain objects
        """
        if not message_ids:
            logger.debug("No message IDs provided")
            return []

        logger.debug(f"Fetching {len(message_ids)} messages by IDs")

        query = select(MessageModel).where(MessageModel.id.in_(message_ids))

        result = await self.session.execute(query)
        models = result.scalars().all()

        logger.debug(f"Found {len(models)} messages")

        return [self._to_domain(model) for model in models]

    def _to_domain(self, model: MessageModel) -> Message:
        """
        Convert SQLAlchemy model to domain object.

        Args:
            model: MessageModel instance

        Returns:
            Message domain object
        """
        return Message(
            id=model.id,
            external_id=model.external_id,
            issue_id=model.issue_id,
            source_id=model.source_id,
            author_type=model.author_type,
            content=model.content,
            created_at=model.created_at,
        )
