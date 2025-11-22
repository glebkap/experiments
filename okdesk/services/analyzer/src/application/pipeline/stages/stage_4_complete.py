"""Stage 4: Complete pipeline execution."""

import logging

from ..pipeline_context import PipelineContext
from .base_stage import BaseStage

logger = logging.getLogger(__name__)


class Stage4Complete(BaseStage):
    """Stage 4: Finalize pipeline execution and log statistics."""

    @property
    def name(self) -> str:
        """Stage name for logging."""
        return "Stage 4: Complete"

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Complete pipeline execution.

        Args:
            context: Pipeline context with final state

        Returns:
            Context with completion statistics
        """
        logger.info("Pipeline execution completed successfully")
        logger.info(f"Statistics: {context.stats}")

        # Add completion timestamp
        from datetime import datetime

        context.add_stat("completed_at", datetime.utcnow().isoformat())

        return context
