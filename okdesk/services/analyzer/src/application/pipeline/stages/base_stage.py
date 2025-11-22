"""Base class for pipeline stages."""

import logging
from abc import ABC, abstractmethod

from ..pipeline_context import PipelineContext

logger = logging.getLogger(__name__)


class BaseStage(ABC):
    """Abstract base class for pipeline stages."""

    @abstractmethod
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute the stage and return updated context.

        Args:
            context: Pipeline context with current state

        Returns:
            Updated pipeline context

        Raises:
            Exception: If stage execution fails
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get stage name for logging.

        Returns:
            Human-readable stage name
        """
        pass

    async def run(self, context: PipelineContext) -> PipelineContext:
        """
        Run stage with logging wrapper.

        Args:
            context: Pipeline context

        Returns:
            Updated context
        """
        logger.info(f"[{self.name}] Starting...")
        try:
            result = await self.execute(context)
            logger.info(f"[{self.name}] Completed successfully")
            return result
        except Exception as e:
            logger.error(f"[{self.name}] Failed: {e}", exc_info=True)
            raise
