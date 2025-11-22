"""Stage 0: Fetch unprocessed issues from database."""

import logging

from ....domain.repositories.issue_repository import IssueRepository
from ..pipeline_context import PipelineContext
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
        # Get unprocessed issues
        issues = await self.issue_repo.get_unprocessed_issues(limit=context.config.batch_size)

        # Update context
        context.issues = issues
        context.add_stat("issues_fetched", len(issues))

        logger.info(f"Fetched {len(issues)} unprocessed issues")

        if not issues:
            logger.warning("No unprocessed issues found")

        return context
