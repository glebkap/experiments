"""Unit tests for domain models."""

import pytest
from datetime import datetime
from uuid import uuid4

from src.domain.models import (
    Source,
    SourceType,
    Issue,
    IssueStatus,
    Message,
    AuthorType,
    ImportJob,
    ImportStatus,
)


class TestSource:
    """Tests for Source entity."""

    def test_create_valid_source(self):
        """Test creating a valid source."""
        source = Source(
            id=uuid4(),
            name="OKDesk Production",
            type=SourceType.OKDESK,
            config={"api_key": "test"},
            created_at=datetime.utcnow(),
        )
        assert source.name == "OKDesk Production"
        assert source.type == SourceType.OKDESK

    def test_empty_name_raises_error(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="Source name cannot be empty"):
            Source(
                id=uuid4(),
                name="",
                type=SourceType.OKDESK,
                config=None,
                created_at=datetime.utcnow(),
            )


class TestIssue:
    """Tests for Issue entity."""

    def test_create_valid_issue(self):
        """Test creating a valid issue."""
        issue = Issue(
            id=uuid4(),
            external_id="12345",
            source_id=uuid4(),
            title="Test Issue",
            description="Description",
            status=IssueStatus.OPENED,
            priority=2,
            created_at=datetime.utcnow(),
            updated_at=None,
            completed_at=None,
        )
        assert issue.external_id == "12345"
        assert issue.priority == 2

    def test_invalid_priority_raises_error(self):
        """Test that invalid priority raises ValueError."""
        with pytest.raises(ValueError, match="priority must be between 1 and 4"):
            Issue(
                id=uuid4(),
                external_id="12345",
                source_id=uuid4(),
                title="Test",
                description=None,
                status=None,
                priority=5,  # Invalid
                created_at=None,
                updated_at=None,
                completed_at=None,
            )


class TestMessage:
    """Tests for Message entity."""

    def test_create_valid_message(self):
        """Test creating a valid message."""
        message = Message(
            id=uuid4(),
            issue_id=uuid4(),
            external_id="msg-123",
            author_id="user-1",
            author_name="John Doe",
            author_type=AuthorType.EMPLOYEE,
            content="Test message content",
            is_public=True,
            published_at=datetime.utcnow(),
        )
        assert message.content == "Test message content"
        assert message.author_type == AuthorType.EMPLOYEE

    def test_empty_content_raises_error(self):
        """Test that empty content raises ValueError."""
        with pytest.raises(ValueError, match="Message content cannot be empty"):
            Message(
                id=uuid4(),
                issue_id=uuid4(),
                external_id="msg-123",
                author_id=None,
                author_name=None,
                author_type=None,
                content="",  # Empty
                is_public=True,
                published_at=None,
            )


class TestImportJob:
    """Tests for ImportJob entity."""

    def test_create_import_job(self):
        """Test creating an import job."""
        job = ImportJob(
            id=uuid4(),
            source_id=uuid4(),
            filename="test.jsonl",
            file_path="/data/test.jsonl",
            started_at=datetime.utcnow(),
            completed_at=None,
            status=ImportStatus.IN_PROGRESS,
            stats=None,
            error_message=None,
        )
        assert job.status == ImportStatus.IN_PROGRESS
        assert job.filename == "test.jsonl"

    def test_mark_completed(self):
        """Test marking import as completed."""
        job = ImportJob(
            id=uuid4(),
            source_id=uuid4(),
            filename="test.jsonl",
            file_path="/data/test.jsonl",
            started_at=datetime.utcnow(),
            completed_at=None,
            status=ImportStatus.IN_PROGRESS,
            stats=None,
            error_message=None,
        )

        stats = {"total": 10, "new": 8}
        job.mark_completed(stats)

        assert job.status == ImportStatus.COMPLETED
        assert job.stats == stats
        assert job.error_message is None
        assert job.completed_at is not None

    def test_mark_failed(self):
        """Test marking import as failed."""
        job = ImportJob(
            id=uuid4(),
            source_id=uuid4(),
            filename="test.jsonl",
            file_path="/data/test.jsonl",
            started_at=datetime.utcnow(),
            completed_at=None,
            status=ImportStatus.IN_PROGRESS,
            stats=None,
            error_message=None,
        )

        job.mark_failed("File not found")

        assert job.status == ImportStatus.FAILED
        assert job.error_message == "File not found"
        assert job.completed_at is not None
