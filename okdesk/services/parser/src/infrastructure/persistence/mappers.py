"""Mappers for converting between domain models and SQLAlchemy models."""

from ...domain.models import ImportJob, Issue, Message, Source
from .models import ImportModel, IssueModel, MessageModel, SourceModel


class SourceMapper:
    """Mapper for Source entity."""

    @staticmethod
    def to_domain(model: SourceModel) -> Source:
        """Convert SQLAlchemy model to domain entity."""
        return Source(
            id=model.id,
            name=model.name,
            type=model.type,
            config=model.config,
            created_at=model.created_at,
        )

    @staticmethod
    def to_model(entity: Source) -> SourceModel:
        """Convert domain entity to SQLAlchemy model."""
        return SourceModel(
            id=entity.id,
            name=entity.name,
            type=entity.type,
            config=entity.config,
            created_at=entity.created_at,
        )


class IssueMapper:
    """Mapper for Issue entity."""

    @staticmethod
    def to_domain(model: IssueModel) -> Issue:
        """Convert SQLAlchemy model to domain entity."""
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
            completed_at=model.completed_at,
        )

    @staticmethod
    def to_model(entity: Issue) -> IssueModel:
        """Convert domain entity to SQLAlchemy model."""
        return IssueModel(
            id=entity.id,
            external_id=entity.external_id,
            source_id=entity.source_id,
            title=entity.title,
            description=entity.description,
            status=entity.status,
            priority=entity.priority,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            completed_at=entity.completed_at,
        )


class MessageMapper:
    """Mapper for Message entity."""

    @staticmethod
    def to_domain(model: MessageModel) -> Message:
        """Convert SQLAlchemy model to domain entity."""
        return Message(
            id=model.id,
            issue_id=model.issue_id,
            external_id=model.external_id,
            author_id=model.author_id,
            author_name=model.author_name,
            author_type=model.author_type,
            content=model.content,
            is_public=model.is_public,
            published_at=model.published_at,
        )

    @staticmethod
    def to_model(entity: Message) -> MessageModel:
        """Convert domain entity to SQLAlchemy model."""
        return MessageModel(
            id=entity.id,
            issue_id=entity.issue_id,
            external_id=entity.external_id,
            author_id=entity.author_id,
            author_name=entity.author_name,
            author_type=entity.author_type,
            content=entity.content,
            is_public=entity.is_public,
            published_at=entity.published_at,
        )


class ImportJobMapper:
    """Mapper for ImportJob entity."""

    @staticmethod
    def to_domain(model: ImportModel) -> ImportJob:
        """Convert SQLAlchemy model to domain entity."""
        return ImportJob(
            id=model.id,
            source_id=model.source_id,
            filename=model.filename,
            file_path=model.file_path,
            started_at=model.started_at,
            completed_at=model.completed_at,
            status=model.status,
            stats=model.stats,
            error_message=model.error_message,
        )

    @staticmethod
    def to_model(entity: ImportJob) -> ImportModel:
        """Convert domain entity to SQLAlchemy model."""
        return ImportModel(
            id=entity.id,
            source_id=entity.source_id,
            filename=entity.filename,
            file_path=entity.file_path,
            started_at=entity.started_at,
            completed_at=entity.completed_at,
            status=entity.status,
            stats=entity.stats,
            error_message=entity.error_message,
        )
