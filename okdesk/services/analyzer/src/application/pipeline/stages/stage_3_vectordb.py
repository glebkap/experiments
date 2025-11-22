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
            logger.warning("No embeddings to store")
            return context

        # Extract data for storage
        issue_ids = [pi.id for pi in context.preprocessed_issues]
        documents = [pi.content for pi in context.preprocessed_issues]

        logger.info(f"Storing {len(issue_ids)} embeddings in ChromaDB...")

        # Save to ChromaDB (synchronous operation)
        self.vectordb.save_embeddings(
            issue_ids=issue_ids, embeddings=context.embeddings, documents=documents
        )

        # Update statistics
        context.add_stat("embeddings_stored", len(issue_ids))

        logger.info(f"Stored {len(issue_ids)} embeddings successfully")

        return context
