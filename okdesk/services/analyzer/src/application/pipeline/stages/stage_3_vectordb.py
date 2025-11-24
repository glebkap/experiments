"""Stage 3: Save embeddings to vector database (ChromaDB)."""

import logging

from ....domain.services.vector_db_service import VectorDBService
from ..pipeline_context import PipelineContext
from .base_stage import BaseStage

logger = logging.getLogger(__name__)


class Stage3VectorDBStorage(BaseStage):
    """Stage 3: Store embeddings in ChromaDB for semantic search."""

    def __init__(self, vectordb: VectorDBService):
        """
        Initialize stage with dependencies.

        Args:
            vectordb: Vector database service
        """
        self.vectordb = vectordb

    @property
    def name(self) -> str:
        """Stage name for logging."""
        return "Stage 3: Vector DB Storage"

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Save embeddings to ChromaDB.

        Args:
            context: Pipeline context with embeddings

        Returns:
            Context with updated statistics
        """
        if context.embeddings is None or not context.preprocessed_issues:
            logger.warning("[Stage 3] No embeddings to store")
            return context

        # Extract data for storage
        issue_ids = [pi.id for pi in context.preprocessed_issues]
        documents = [pi.content for pi in context.preprocessed_issues]

        logger.info(f"[Stage 3] Storing {len(issue_ids)} embeddings in ChromaDB...")

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[Stage 3] First 3 issue IDs: {[str(id) for id in issue_ids[:3]]}")
            logger.debug(
                f"[Stage 3] Embedding shape: {context.embeddings.shape}, "
                f"dtype: {context.embeddings.dtype}"
            )

        # Save to ChromaDB (asynchronous operation)
        import time
        start_time = time.time()

        try:
            await self.vectordb.save_embeddings(
                issue_ids=issue_ids, embeddings=context.embeddings, documents=documents
            )
            elapsed = time.time() - start_time

            # Update statistics
            context.add_stat("embeddings_stored", len(issue_ids))
            context.add_stat("vectordb_time_seconds", round(elapsed, 2))

            logger.info(
                f"[Stage 3] Stored {len(issue_ids)} embeddings successfully "
                f"(time={elapsed:.2f}s, rate={len(issue_ids)/elapsed:.1f} issues/sec)"
            )

            if logger.isEnabledFor(logging.DEBUG):
                # Verify storage by checking collection count
                total_count = self.vectordb.count_total()
                logger.debug(f"[Stage 3] Total embeddings in ChromaDB: {total_count}")

        except Exception as e:
            logger.error(f"[Stage 3] Failed to store embeddings: {e}", exc_info=True)
            raise

        return context
