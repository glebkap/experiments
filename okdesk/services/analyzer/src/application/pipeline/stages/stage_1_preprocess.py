"""Stage 1: Preprocess issues (clean, normalize, lemmatize)."""

import logging
from datetime import datetime

from ....domain.models.preprocessed_issue import PreprocessedIssue
from ....domain.repositories.preprocessed_issue_repository import PreprocessedIssueRepository
from ....domain.services.text_preprocessor import TextPreprocessor
from ..pipeline_context import PipelineContext
from .base_stage import BaseStage

logger = logging.getLogger(__name__)


class Stage1Preprocess(BaseStage):
    """Stage 1: Preprocess issue text (HTML cleanup, normalization, lemmatization)."""

    def __init__(
        self, preprocessor: TextPreprocessor, preprocessed_repo: PreprocessedIssueRepository
    ):
        """
        Initialize stage with dependencies.

        Args:
            preprocessor: Text preprocessing service
            preprocessed_repo: Repository for saving preprocessed issues
        """
        self.preprocessor = preprocessor
        self.preprocessed_repo = preprocessed_repo

    @property
    def name(self) -> str:
        """Stage name for logging."""
        return "Stage 1: Preprocess"

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Preprocess issues: clean HTML, normalize, lemmatize.

        Args:
            context: Pipeline context with fetched issues

        Returns:
            Context with preprocessed issues
        """
        if not context.issues:
            logger.warning("No issues to preprocess")
            return context

        preprocessed = []
        skipped_empty = 0

        for issue in context.issues:
            # Preprocess title + description
            content = self.preprocessor.preprocess_issue(issue.title, issue.description)

            # Skip empty content
            if not content or not content.strip():
                skipped_empty += 1
                logger.debug(f"Skipping issue {issue.id} - empty content after preprocessing")
                continue

            preprocessed.append(
                PreprocessedIssue(
                    id=issue.id, content=content, processed_at=datetime.utcnow()
                )
            )

        # Batch save to database
        if preprocessed:
            await self.preprocessed_repo.save_batch(preprocessed)

        # Update context
        context.preprocessed_issues = preprocessed
        context.add_stat("issues_preprocessed", len(preprocessed))
        context.add_stat("issues_skipped_empty", skipped_empty)

        logger.info(
            f"Preprocessed {len(preprocessed)} issues "
            f"(skipped {skipped_empty} with empty content)"
        )

        return context
