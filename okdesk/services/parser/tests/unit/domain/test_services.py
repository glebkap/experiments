"""Unit tests for domain services."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from src.domain.models import Issue, Message
from src.domain.services import DeduplicationService, ImportService


class TestDeduplicationService:
    """Tests for DeduplicationService."""

    @pytest.mark.asyncio
    async def test_is_issue_duplicate_true(self):
        """Test checking for duplicate issue when exists."""
        issue_repo = AsyncMock()
        message_repo = AsyncMock()
        issue_repo.get_by_external_id.return_value = MagicMock()  # Exists

        service = DeduplicationService(issue_repo, message_repo)
        result = await service.is_issue_duplicate("ext-123", uuid4())

        assert result is True

    @pytest.mark.asyncio
    async def test_is_issue_duplicate_false(self):
        """Test checking for duplicate issue when not exists."""
        issue_repo = AsyncMock()
        message_repo = AsyncMock()
        issue_repo.get_by_external_id.return_value = None  # Not exists

        service = DeduplicationService(issue_repo, message_repo)
        result = await service.is_issue_duplicate("ext-123", uuid4())

        assert result is False


class TestImportService:
    """Tests for ImportService."""

    def test_calculate_stats(self):
        """Test calculating import statistics."""
        import_repo = AsyncMock()
        issue_repo = AsyncMock()
        message_repo = AsyncMock()
        dedup_service = MagicMock()

        service = ImportService(import_repo, issue_repo, message_repo, dedup_service)

        stats = service.calculate_stats(
            total_issues=10,
            new_issues=7,
            total_messages=50,
            new_messages=40,
        )

        assert stats["total_issues"] == 10
        assert stats["new_issues"] == 7
        assert stats["updated_issues"] == 3
        assert stats["total_messages"] == 50
        assert stats["new_messages"] == 40
        assert stats["updated_messages"] == 10
