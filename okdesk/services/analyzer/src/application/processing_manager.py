"""Processing manager for background issue processing."""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Optional

from .use_cases.process_issues_batch import ProcessIssuesBatchUseCase

logger = logging.getLogger(__name__)


class ProcessingState(str, Enum):
    """Processing state enum."""

    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"


class ProcessingManager:
    """
    Manager for background processing of issues.

    Handles start/pause/stop operations and automatic polling for new issues.
    """

    def __init__(
        self,
        process_batch_use_case: ProcessIssuesBatchUseCase,
        batch_size: int = 100,
        poll_interval_seconds: float = 1.0,
        device: str = "cpu",
    ):
        """
        Initialize processing manager.

        Args:
            process_batch_use_case: Use case for batch processing
            batch_size: Number of issues to process in one batch
            poll_interval_seconds: Interval between polls for new issues
            device: Device for embeddings (cpu/cuda/mps/auto)
        """
        self.process_batch_use_case = process_batch_use_case
        self.batch_size = batch_size
        self.poll_interval_seconds = poll_interval_seconds
        self.device = device

        self.state = ProcessingState.STOPPED
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        # Statistics
        self.total_processed = 0
        self.total_batches = 0
        self.started_at: Optional[datetime] = None
        self.last_batch_at: Optional[datetime] = None

    def get_status(self) -> dict:
        """
        Get current processing status.

        Returns:
            Dictionary with status information
        """
        uptime_seconds = None
        if self.started_at:
            uptime_seconds = (datetime.utcnow() - self.started_at).total_seconds()

        return {
            "state": self.state.value,
            "batch_size": self.batch_size,
            "poll_interval_seconds": self.poll_interval_seconds,
            "device": self.device,
            "total_processed": self.total_processed,
            "total_batches": self.total_batches,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_batch_at": self.last_batch_at.isoformat() if self.last_batch_at else None,
            "uptime_seconds": uptime_seconds,
        }

    async def start(self) -> dict:
        """
        Start background processing.

        Returns:
            Status dictionary

        Raises:
            RuntimeError: If already running
        """
        if self.state == ProcessingState.RUNNING:
            raise RuntimeError("Processing is already running")

        logger.info("Starting background processing")

        self.state = ProcessingState.RUNNING
        self.started_at = datetime.utcnow()
        self._stop_event.clear()

        # Create background task
        self._task = asyncio.create_task(self._processing_loop())

        return self.get_status()

    async def pause(self) -> dict:
        """
        Pause background processing.

        Returns:
            Status dictionary

        Raises:
            RuntimeError: If not running
        """
        if self.state != ProcessingState.RUNNING:
            raise RuntimeError("Processing is not running")

        logger.info("Pausing background processing")
        self.state = ProcessingState.PAUSED

        return self.get_status()

    async def resume(self) -> dict:
        """
        Resume background processing after pause.

        Returns:
            Status dictionary

        Raises:
            RuntimeError: If not paused
        """
        if self.state != ProcessingState.PAUSED:
            raise RuntimeError("Processing is not paused")

        logger.info("Resuming background processing")
        self.state = ProcessingState.RUNNING

        return self.get_status()

    async def stop(self) -> dict:
        """
        Stop background processing.

        Returns:
            Final status dictionary
        """
        if self.state == ProcessingState.STOPPED:
            logger.warning("Processing is already stopped")
            return self.get_status()

        logger.info("Stopping background processing")

        self.state = ProcessingState.STOPPED
        self._stop_event.set()

        # Wait for background task to finish
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("Processing task did not stop gracefully, cancelling")
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass

            self._task = None

        return self.get_status()

    async def process_single_batch(self) -> dict:
        """
        Process a single batch manually (regardless of state).

        Returns:
            Processing result dictionary
        """
        logger.info("Processing single batch manually")

        result = await self.process_batch_use_case.execute(
            batch_size=self.batch_size, device=self.device
        )

        self.total_processed += result.processed_count
        self.total_batches += 1
        self.last_batch_at = datetime.utcnow()

        return {
            "success": result.success,
            "processed_count": result.processed_count,
            "duration_seconds": result.duration_seconds,
            "stats": result.stats,
            "errors": result.errors,
        }

    async def _processing_loop(self):
        """
        Background processing loop.

        Polls for new issues every poll_interval_seconds and processes them.
        """
        logger.info("Processing loop started")

        try:
            while not self._stop_event.is_set():
                # Check if paused
                if self.state == ProcessingState.PAUSED:
                    await asyncio.sleep(0.5)
                    continue

                # Check if stopped
                if self.state == ProcessingState.STOPPED:
                    break

                # Process batch
                try:
                    logger.info(
                        f"[Batch #{self.total_batches + 1}] Starting processing "
                        f"(batch_size={self.batch_size}, device={self.device}, total_processed={self.total_processed})"
                    )

                    result = await self.process_batch_use_case.execute(
                        batch_size=self.batch_size, device=self.device
                    )

                    if result.processed_count > 0:
                        self.total_processed += result.processed_count
                        self.total_batches += 1
                        self.last_batch_at = datetime.utcnow()

                        logger.info(
                            f"[Batch #{self.total_batches}] ✅ Successfully processed {result.processed_count} issues "
                            f"in {result.duration_seconds:.2f}s (total: {self.total_processed} issues)"
                        )

                        # Log stats if available
                        if result.stats:
                            logger.info(f"[Batch #{self.total_batches}] Stats: {result.stats}")

                        # Log errors if any
                        if result.errors:
                            logger.warning(f"[Batch #{self.total_batches}] Errors occurred: {result.errors}")
                    else:
                        # No issues to process, wait before polling again
                        logger.debug("No unprocessed issues found, waiting for next poll...")

                except Exception as e:
                    logger.error(f"[Batch #{self.total_batches + 1}] ❌ Error in processing loop: {e}", exc_info=True)

                # Wait before next poll
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self.poll_interval_seconds
                    )
                except asyncio.TimeoutError:
                    pass  # Normal timeout, continue loop

        except asyncio.CancelledError:
            logger.info("Processing loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Processing loop failed: {e}", exc_info=True)
            self.state = ProcessingState.STOPPED
        finally:
            logger.info("Processing loop stopped")


# Global singleton instance
_processing_manager: Optional[ProcessingManager] = None


def set_processing_manager(manager: ProcessingManager) -> None:
    """
    Set global processing manager instance.

    Args:
        manager: ProcessingManager instance to register
    """
    global _processing_manager
    _processing_manager = manager
    logger.info("ProcessingManager registered globally")


def get_processing_manager() -> Optional[ProcessingManager]:
    """
    Get global processing manager instance.

    Returns:
        ProcessingManager instance or None if not initialized
    """
    return _processing_manager
