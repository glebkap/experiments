"""Service for checking duplicates of issues and messages."""

from uuid import UUID

from ..repositories import IssueRepository, MessageRepository


class DeduplicationService:
    """Service for deduplication logic."""

    def __init__(
        self,
        issue_repository: IssueRepository,
        message_repository: MessageRepository,
    ) -> None:
        """Initialize deduplication service."""
        self._issue_repo = issue_repository
        self._message_repo = message_repository

    async def is_issue_duplicate(self, external_id: str, source_id: UUID) -> bool:
        """
        Check if issue already exists.

        Args:
            external_id: External issue identifier
            source_id: Source identifier

        Returns:
            True if issue exists, False otherwise
        """
        existing = await self._issue_repo.get_by_external_id(external_id, source_id)
        return existing is not None

    async def is_message_duplicate(self, external_id: str, issue_id: UUID) -> bool:
        """
        Check if message already exists.

        Args:
            external_id: External message identifier
            issue_id: Issue identifier

        Returns:
            True if message exists, False otherwise
        """
        existing = await self._message_repo.get_by_external_id(external_id, issue_id)
        return existing is not None
