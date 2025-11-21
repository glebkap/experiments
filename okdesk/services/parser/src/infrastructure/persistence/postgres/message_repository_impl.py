"""PostgreSQL implementation of MessageRepository."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ....domain.models import Message
from ....domain.repositories import MessageRepository
from ..mappers import MessageMapper
from ..models import MessageModel


class MessageRepositoryImpl(MessageRepository):
    """PostgreSQL implementation of MessageRepository."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session."""
        self._session = session
        self._mapper = MessageMapper()

    async def get_by_id(self, message_id: UUID) -> Message | None:
        """Get message by ID."""
        result = await self._session.execute(
            select(MessageModel).where(MessageModel.id == message_id)
        )
        model = result.scalar_one_or_none()
        return self._mapper.to_domain(model) if model else None

    async def get_by_external_id(self, external_id: str, issue_id: UUID) -> Message | None:
        """Get message by external_id and issue_id."""
        result = await self._session.execute(
            select(MessageModel).where(
                MessageModel.external_id == external_id,
                MessageModel.issue_id == issue_id,
            )
        )
        model = result.scalar_one_or_none()
        return self._mapper.to_domain(model) if model else None

    async def get_by_issue_id(self, issue_id: UUID) -> list[Message]:
        """Get all messages for an issue."""
        result = await self._session.execute(
            select(MessageModel).where(MessageModel.issue_id == issue_id)
        )
        models = result.scalars().all()
        return [self._mapper.to_domain(model) for model in models]

    async def create(self, message: Message) -> Message:
        """Create a new message."""
        model = self._mapper.to_model(message)
        self._session.add(model)
        await self._session.flush()
        await self._session.refresh(model)
        return self._mapper.to_domain(model)

    async def update(self, message: Message) -> Message:
        """Update an existing message."""
        result = await self._session.execute(
            select(MessageModel).where(MessageModel.id == message.id)
        )
        model = result.scalar_one_or_none()
        if not model:
            raise ValueError(f"Message with id {message.id} not found")

        # Update fields
        model.issue_id = message.issue_id
        model.external_id = message.external_id
        model.author_id = message.author_id
        model.author_name = message.author_name
        model.author_type = message.author_type
        model.content = message.content
        model.is_public = message.is_public
        model.published_at = message.published_at

        await self._session.flush()
        await self._session.refresh(model)
        return self._mapper.to_domain(model)

    async def bulk_create(self, messages: list[Message]) -> list[Message]:
        """Create multiple messages in bulk."""
        models = [self._mapper.to_model(msg) for msg in messages]
        self._session.add_all(models)
        await self._session.flush()

        # Refresh all models to get generated IDs
        for model in models:
            await self._session.refresh(model)

        return [self._mapper.to_domain(model) for model in models]
