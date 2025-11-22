"""Stage 2: Generate embeddings for preprocessed issues."""

import logging

from ....domain.services.embedding_generator import EmbeddingGenerator
from ..pipeline_context import PipelineContext
from .base_stage import BaseStage

logger = logging.getLogger(__name__)


class Stage2GenerateEmbeddings(BaseStage):
    """Stage 2: Generate vector embeddings using SentenceTransformers."""

    def __init__(self, embedding_gen: EmbeddingGenerator):
        """
        Initialize stage with dependencies.

        Args:
            embedding_gen: Embedding generation service
        """
        self.embedding_gen = embedding_gen

    @property
    def name(self) -> str:
        """Stage name for logging."""
        return "Stage 2: Generate Embeddings"

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Generate embeddings for preprocessed issues.

        Args:
            context: Pipeline context with preprocessed issues

        Returns:
            Context with generated embeddings
        """
        if not context.preprocessed_issues:
            logger.warning("No preprocessed issues for embedding generation")
            return context

        # Extract preprocessed texts
        texts = [pi.content for pi in context.preprocessed_issues]

        logger.info(f"Generating embeddings for {len(texts)} texts...")

        # Generate embeddings (synchronous operation)
        embeddings = self.embedding_gen.generate_batch(
            texts,
            batch_size=context.config.embedding_batch_size,
            show_progress=True,
        )

        # Update context
        context.embeddings = embeddings
        context.add_stat("embeddings_generated", len(embeddings))
        context.add_stat("embedding_dimension", self.embedding_gen.get_dimension())

        logger.info(
            f"Generated {len(embeddings)} embeddings "
            f"(dim={self.embedding_gen.get_dimension()})"
        )

        return context
