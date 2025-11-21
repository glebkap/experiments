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
    ) -> tuple[Issue, str]:
        """
        Process an issue - create or update.

        Args:
            issue: Issue to process

        Returns:
            Tuple of (processed issue, status) where status is 'created', 'updated', or 'unchanged'
        """
        # Check if issue exists by external_id (across all sources)
        existing = await self._issue_repo.find_by_external_id(issue.external_id)

        if existing:
            # Check if this is the same issue by comparing title and description
            is_same_issue = (
                existing.title == issue.title
                and existing.description == issue.description
            )

            if not is_same_issue:
                # Different issue with same external_id - this should not happen normally
                logger.warning(
                    f"Issue {issue.external_id} has different title/description than existing one. "
                    f"Creating duplicate with different content."
                )
                created = await self._issue_repo.create(issue)
                return created, 'created'

            # Same issue - check if other fields changed
            content_changed = (
                existing.status != issue.status
                or existing.priority != issue.priority
            )

            if content_changed:
                logger.debug(f"Issue {issue.external_id} status/priority changed, updating")
                issue.id = existing.id
                # Keep existing source_id to avoid changing source
                issue.source_id = existing.source_id
                updated = await self._issue_repo.update(issue)
                return updated, 'updated'
            else:
                logger.debug(f"Issue {issue.external_id} unchanged, skipping update")
                return existing, 'unchanged'

        logger.debug(f"Creating new issue {issue.external_id}")
        created = await self._issue_repo.create(issue)
        return created, 'created'

    async def process_message(
        self,
        message: Message,
    ) -> tuple[Message, str]:
        """
        Process a message - create or update.

        Args:
            message: Message to process

        Returns:
            Tuple of (processed message, status) where status is 'created', 'updated', or 'unchanged'
        """
        is_duplicate = await self._dedup.is_message_duplicate(
            message.external_id,
            message.issue_id,
        )

        if is_duplicate:
            existing = await self._message_repo.get_by_external_id(
                message.external_id,
                message.issue_id,
            )
            if existing:
                # Check if content actually changed
                content_changed = (
                    existing.content != message.content
                    or existing.author_name != message.author_name
                    or existing.is_public != message.is_public
                )

                if content_changed:
                    logger.debug(f"Message {message.external_id} content changed, updating")
                    message.id = existing.id
                    updated = await self._message_repo.update(message)
                    return updated, 'updated'
                else:
                    logger.debug(f"Message {message.external_id} unchanged, skipping update")
                    return existing, 'unchanged'

        logger.debug(f"Creating new message {message.external_id}")
        created = await self._message_repo.create(message)
        return created, 'created'

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
