"""Stage 0: Fetch unprocessed issues from database."""

import logging

from ....config import settings
from ....domain.repositories.issue_repository import IssueRepository
from ..pipeline_context import PipelineContext
from ..utils.logging_helpers import format_issue_debug_info
from .base_stage import BaseStage

logger = logging.getLogger(__name__)


class Stage0FetchIssues(BaseStage):
    """Stage 0: Fetch unprocessed issues from PostgreSQL."""

    def __init__(self, issue_repo: IssueRepository):
        """
        Initialize stage with dependencies.

        Args:
            issue_repo: Issue repository for database access
        """
        self.issue_repo = issue_repo

    @property
    def name(self) -> str:
        """Stage name for logging."""
        return "Stage 0: Fetch Issues"

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Fetch unprocessed issues from database.

        Issues are considered unprocessed if they don't exist in preprocessed_issues table.

        Args:
            context: Pipeline context

        Returns:
            Context with fetched issues
        """
        logger.info(f"[Stage 0] Fetching up to {context.config.batch_size} unprocessed issues...")

        # Get unprocessed issues
        issues = await self.issue_repo.get_unprocessed_issues(limit=context.config.batch_size)

        # Update context
        context.issues = issues
        context.add_stat("issues_fetched", len(issues))

        if not issues:
            logger.warning("[Stage 0] No unprocessed issues found")
        else:
            logger.info(f"[Stage 0] Fetched {len(issues)} unprocessed issues")

            # Debug: log detailed information about first few issues
            if logger.isEnabledFor(logging.DEBUG):
                # Log all issue IDs
                issue_ids = [str(issue.id) for issue in issues]
                logger.debug(f"[Stage 0] Issue IDs ({len(issue_ids)} total):")
                logger.debug(f"    {', '.join(issue_ids[:10])}{'...' if len(issue_ids) > 10 else ''}")

                # Log detailed info for first N issues
                sample_size = min(settings.debug_log_sample_size, len(issues))
                logger.debug(f"[Stage 0] Detailed info for first {sample_size} issues:")

                for idx, issue in enumerate(issues[:sample_size], 1):
                    debug_info = format_issue_debug_info(
                        issue_id=str(issue.id),
                        title=issue.title,
                        description=issue.description,
                        status=issue.status,
                        index=idx,
                        total=sample_size,
                    )
                    logger.debug(f"\n{debug_info}\n")

        return context
