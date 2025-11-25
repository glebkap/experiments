"""SQLAlchemy models for database tables."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import (
    TIMESTAMP,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


class IssueModel(Base):
    """SQLAlchemy model for issues table."""

    __tablename__ = "issues"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    external_id: Mapped[str] = mapped_column(String(255), nullable=False)
    source_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("sources.id"), nullable=False
    )
    title: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    priority: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )

    # Relationships
    preprocessed: Mapped[Optional["PreprocessedIssueModel"]] = relationship(
        "PreprocessedIssueModel", back_populates="issue", uselist=False
    )
    messages: Mapped[list["MessageModel"]] = relationship(
        "MessageModel", back_populates="issue"
    )
    cluster_assignments: Mapped[list["IssueClusterModel"]] = relationship(
        "IssueClusterModel", back_populates="issue"
    )

    __table_args__ = (Index("idx_issues_source_external", "source_id", "external_id"),)


class PreprocessedIssueModel(Base):
    """SQLAlchemy model for preprocessed_issues table."""

    __tablename__ = "preprocessed_issues"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("issues.id"), primary_key=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    processed_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    issue: Mapped["IssueModel"] = relationship(
        "IssueModel", back_populates="preprocessed"
    )

    __table_args__ = (
        Index(
            "idx_preprocessed_issues_fts",
            "content",
            postgresql_using="gin",
            postgresql_ops={"content": "gin_trgm_ops"},
        ),
    )


class ClusterModel(Base):
    """SQLAlchemy model for clusters table."""

    __tablename__ = "clusters"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    cluster_label: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    centroid_embedding: Mapped[list[float]] = mapped_column(
        ARRAY(Float), nullable=False
    )
    size: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    issue_clusters: Mapped[list["IssueClusterModel"]] = relationship(
        "IssueClusterModel", back_populates="cluster"
    )


class MessageModel(Base):
    """SQLAlchemy model for messages table."""

    __tablename__ = "messages"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    external_id: Mapped[str] = mapped_column(String(255), nullable=False)
    issue_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("issues.id"), nullable=False
    )
    author_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    author_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    author_type: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    is_public: Mapped[bool] = mapped_column(default=True, nullable=False)
    published_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )

    # Relationships
    issue: Mapped["IssueModel"] = relationship("IssueModel", back_populates="messages")

    __table_args__ = (Index("idx_messages_issue", "issue_id"),)


class IssueClusterModel(Base):
    """SQLAlchemy model for issue_clusters table."""

    __tablename__ = "issue_clusters"

    issue_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("issues.id"), primary_key=True
    )
    cluster_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("clusters.id"), nullable=False
    )
    distance_to_centroid: Mapped[float] = mapped_column(nullable=False)
    assigned_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    issue: Mapped["IssueModel"] = relationship(
        "IssueModel", back_populates="cluster_assignments"
    )
    cluster: Mapped["ClusterModel"] = relationship(
        "ClusterModel", back_populates="issue_clusters"
    )

    __table_args__ = (Index("idx_issue_clusters_cluster", "cluster_id"),)
