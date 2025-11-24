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
        from datetime import datetime

        logger.info("[Stage 4] ========== Pipeline Execution Completed ==========")

        # Add completion timestamp
        context.add_stat("completed_at", datetime.utcnow().isoformat())

        # Log comprehensive statistics
        logger.info("[Stage 4] Final Statistics:")
        logger.info(f"  - Issues fetched: {context.stats.get('issues_fetched', 0)}")
        logger.info(f"  - Issues preprocessed: {context.stats.get('issues_preprocessed', 0)}")
        logger.info(f"  - Issues skipped (empty): {context.stats.get('issues_skipped_empty', 0)}")
        logger.info(f"  - Embeddings generated: {context.stats.get('embeddings_generated', 0)}")
        logger.info(f"  - Embeddings stored: {context.stats.get('embeddings_stored', 0)}")
        logger.info(f"  - Embedding dimension: {context.stats.get('embedding_dimension', 0)}")

        # Performance metrics
        if "embedding_time_seconds" in context.stats:
            logger.info(f"  - Embedding time: {context.stats['embedding_time_seconds']}s")
        if "vectordb_time_seconds" in context.stats:
            logger.info(f"  - VectorDB time: {context.stats['vectordb_time_seconds']}s")

        # Calculate total processing rate
        total_processed = context.stats.get('embeddings_stored', 0)
        total_time = context.stats.get('embedding_time_seconds', 0) + context.stats.get('vectordb_time_seconds', 0)
        if total_time > 0:
            logger.info(f"  - Overall rate: {total_processed/total_time:.1f} issues/sec")

        logger.info("[Stage 4] ================================================")

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[Stage 4] Full context stats: {context.stats}")

        return context
