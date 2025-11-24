"""Stage 1: Preprocess issues (clean, normalize, lemmatize)."""

import logging
from datetime import datetime

from ....config import settings
from ....domain.models.preprocessed_issue import PreprocessedIssue
from ....domain.repositories.preprocessed_issue_repository import PreprocessedIssueRepository
from ....domain.services.text_preprocessor import TextPreprocessor
from ..pipeline_context import PipelineContext
from ..utils.logging_helpers import format_preprocessed_content_debug
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
            logger.warning("[Stage 1] No issues to preprocess")
            return context

        logger.info(f"[Stage 1] Preprocessing {len(context.issues)} issues...")

        preprocessed = []
        skipped_empty = 0

        # Determine how many issues to log in detail
        sample_size = min(settings.debug_log_sample_size, len(context.issues))

        for idx, issue in enumerate(context.issues, 1):
            # Preprocess title + description
            content = self.preprocessor.preprocess_issue(issue.title, issue.description)

            # Skip empty content
            if not content or not content.strip():
                skipped_empty += 1
                logger.debug(f"[Stage 1] Skipping issue {issue.id} - empty content after preprocessing")
                continue

            # Log detailed info for first N issues
            if logger.isEnabledFor(logging.DEBUG) and idx <= sample_size:
                debug_info = format_preprocessed_content_debug(
                    issue_id=str(issue.id),
                    original_title_len=len(issue.title or ""),
                    original_desc_len=len(issue.description or ""),
                    preprocessed_content=content,
                    index=idx,
                    total=sample_size,
                )
                logger.debug(f"\n{debug_info}\n")

            preprocessed.append(
                PreprocessedIssue(
                    id=issue.id, content=content, processed_at=datetime.utcnow()
                )
            )

        # Batch save to database
        if preprocessed:
            logger.debug(f"[Stage 1] Saving {len(preprocessed)} preprocessed issues to database...")
            await self.preprocessed_repo.save_batch(preprocessed)
            logger.debug(f"[Stage 1] Successfully saved preprocessed issues")

        # Update context
        context.preprocessed_issues = preprocessed
        context.add_stat("issues_preprocessed", len(preprocessed))
        context.add_stat("issues_skipped_empty", skipped_empty)

        logger.info(
            f"[Stage 1] Preprocessed {len(preprocessed)} issues "
            f"(skipped {skipped_empty} with empty content)"
        )

        return context
