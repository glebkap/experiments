"""Pipeline executor orchestrating all stages."""

import asyncio
import logging
import time
from typing import List

from .pipeline_config import PipelineConfig
from .pipeline_context import PipelineContext
from .stages.base_stage import BaseStage

logger = logging.getLogger(__name__)


class PipelineResult:
    """Result of pipeline execution."""

    def __init__(
        self,
        processed_count: int,
        duration_seconds: float,
        stats: dict,
        errors: List[str],
        success: bool,
    ):
        """
        Initialize pipeline result.

        Args:
            processed_count: Number of issues successfully processed
            duration_seconds: Total execution time
            stats: Detailed statistics
            errors: List of error messages
            success: Whether pipeline completed successfully
        """
        self.processed_count = processed_count
        self.duration_seconds = duration_seconds
        self.stats = stats
        self.errors = errors
        self.success = success

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "processed_count": self.processed_count,
            "duration_seconds": self.duration_seconds,
            "stats": self.stats,
            "errors": self.errors,
            "success": self.success,
        }


class PipelineExecutor:
    """Executor for running pipeline stages sequentially."""

    def __init__(self, config: PipelineConfig, stages: List[BaseStage]):
        """
        Initialize executor with configuration and stages.

        Args:
            config: Pipeline configuration
            stages: List of stages to execute in order
        """
        self.config = config
        self.stages = stages

    async def execute(self) -> PipelineResult:
        """
        Execute complete pipeline.

        Returns:
            Pipeline execution result

        Raises:
            Exception: If pipeline fails and max retries exceeded
        """
        start_time = time.time()
        context = PipelineContext(config=self.config)
        errors = []

        logger.info("=" * 60)
        logger.info("Starting pipeline execution")
        logger.info(f"Configuration: batch_size={self.config.batch_size}")
        logger.info("=" * 60)

        try:
            for stage in self.stages:
                context = await self._execute_stage_with_retry(stage, context)

            success = True

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            errors.append(str(e))
            success = False

        duration = time.time() - start_time

        result = PipelineResult(
            processed_count=context.get_stat("issues_preprocessed", 0),
            duration_seconds=duration,
            stats=context.stats,
            errors=errors,
            success=success,
        )

        logger.info("=" * 60)
        logger.info(f"Pipeline completed in {duration:.2f}s")
        logger.info(f"Success: {success}")
        logger.info(f"Processed: {result.processed_count} issues")
        logger.info("=" * 60)

        return result

    async def _execute_stage_with_retry(
        self, stage: BaseStage, context: PipelineContext
    ) -> PipelineContext:
        """
        Execute stage with retry logic.

        Args:
            stage: Stage to execute
            context: Current pipeline context

        Returns:
            Updated context

        Raises:
            Exception: If all retries fail
        """
        for attempt in range(self.config.max_retries):
            try:
                return await stage.run(context)

            except Exception as e:
                logger.warning(
                    f"[{stage.name}] Attempt {attempt + 1}/{self.config.max_retries} failed: {e}"
                )

                if attempt < self.config.max_retries - 1:
                    logger.info(
                        f"[{stage.name}] Retrying in {self.config.retry_delay_seconds}s..."
                    )
                    await asyncio.sleep(self.config.retry_delay_seconds)
                else:
                    logger.error(f"[{stage.name}] All retries exhausted")
                    raise

        # Should never reach here
        raise RuntimeError(f"Unexpected error in stage {stage.name}")
