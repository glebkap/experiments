"""SQLAlchemy ORM models mapping to database tables."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, Enum as SQLEnum, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from ...domain.models import AuthorType, ImportStatus, IssueStatus, SourceType
from .database import Base


class SourceModel(Base):
    """SQLAlchemy model for sources table."""

    __tablename__ = "sources"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    type: Mapped[SourceType] = mapped_column(
        SQLEnum(SourceType, name="source_type", native_enum=True, values_callable=lambda x: [e.value for e in x]),
        nullable=False
    )
    config: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )


class IssueModel(Base):
    """SQLAlchemy model for issues table."""

    __tablename__ = "issues"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    external_id: Mapped[str] = mapped_column(String(255), nullable=False)
    source_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[IssueStatus | None] = mapped_column(
        SQLEnum(IssueStatus, name="issue_status", native_enum=True, values_callable=lambda x: [e.value for e in x]),
        nullable=True
    )
    priority: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class MessageModel(Base):
    """SQLAlchemy model for messages table."""

    __tablename__ = "messages"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    issue_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    external_id: Mapped[str] = mapped_column(String(255), nullable=False)
    author_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    author_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    author_type: Mapped[AuthorType | None] = mapped_column(
        SQLEnum(AuthorType, name="author_type", native_enum=True, values_callable=lambda x: [e.value for e in x]),
        nullable=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    is_public: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ImportModel(Base):
    """SQLAlchemy model for imports table."""

    __tablename__ = "imports"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    source_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    filename: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[ImportStatus] = mapped_column(
        SQLEnum(ImportStatus, name="import_status", native_enum=True, values_callable=lambda x: [e.value for e in x]),
        nullable=False
    )
    stats: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
