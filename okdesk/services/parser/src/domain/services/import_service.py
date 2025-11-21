"""Domain service for import coordination."""

import logging
from typing import Any
from uuid import UUID

from ..models import ImportJob, ImportStatus, Issue, Message
from ..repositories import ImportRepository, IssueRepository, MessageRepository
from .deduplication_service import DeduplicationService

logger = logging.getLogger(__name__)


class ImportService:
    """Domain service coordinating import process."""

    def __init__(
        self,
        import_repository: ImportRepository,
        issue_repository: IssueRepository,
        message_repository: MessageRepository,
        deduplication_service: DeduplicationService,
    ) -> None:
        """Initialize import service."""
        self._import_repo = import_repository
        self._issue_repo = issue_repository
        self._message_repo = message_repository
        self._dedup = deduplication_service

    async def create_import_job(self, import_job: ImportJob) -> ImportJob:
        """Create a new import job."""
        return await self._import_repo.create(import_job)

    async def update_import_job(self, import_job: ImportJob) -> ImportJob:
        """Update import job status."""
        return await self._import_repo.update(import_job)

    async def process_issue(
        self,
        issue: Issue,
    ) -> tuple[Issue, bool]:
        """
        Process an issue - create or update.

        Args:
            issue: Issue to process

        Returns:
            Tuple of (processed issue, is_new)
        """
        is_duplicate = await self._dedup.is_issue_duplicate(
            issue.external_id,
            issue.source_id,
        )

        if is_duplicate:
            logger.debug(f"Issue {issue.external_id} already exists, updating")
            existing = await self._issue_repo.get_by_external_id(
                issue.external_id,
                issue.source_id,
            )
            if existing:
                # Update existing issue with new data
                issue.id = existing.id
                updated = await self._issue_repo.update(issue)
                return updated, False

        logger.debug(f"Creating new issue {issue.external_id}")
        created = await self._issue_repo.create(issue)
        return created, True

    async def process_message(
        self,
        message: Message,
    ) -> tuple[Message, bool]:
        """
        Process a message - create or update.

        Args:
            message: Message to process

        Returns:
            Tuple of (processed message, is_new)
        """
        is_duplicate = await self._dedup.is_message_duplicate(
            message.external_id,
            message.issue_id,
        )

        if is_duplicate:
            logger.debug(f"Message {message.external_id} already exists, updating")
            existing = await self._message_repo.get_by_external_id(
                message.external_id,
                message.issue_id,
            )
            if existing:
                # Update existing message with new data
                message.id = existing.id
                updated = await self._message_repo.update(message)
                return updated, False

        logger.debug(f"Creating new message {message.external_id}")
        created = await self._message_repo.create(message)
        return created, True

    def calculate_stats(
        self,
        total_issues: int,
        new_issues: int,
        total_messages: int,
        new_messages: int,
    ) -> dict[str, Any]:
        """
        Calculate import statistics.

        Args:
            total_issues: Total number of issues processed
            new_issues: Number of new issues created
            total_messages: Total number of messages processed
            new_messages: Number of new messages created

        Returns:
            Statistics dictionary
        """
        return {
            "total_issues": total_issues,
            "new_issues": new_issues,
            "updated_issues": total_issues - new_issues,
            "total_messages": total_messages,
            "new_messages": new_messages,
            "updated_messages": total_messages - new_messages,
        }
