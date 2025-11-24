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
            logger.warning("[Stage 2] No preprocessed issues for embedding generation")
            return context

        # Extract preprocessed texts
        texts = [pi.content for pi in context.preprocessed_issues]

        logger.info(
            f"[Stage 2] Generating embeddings for {len(texts)} texts "
            f"(batch_size={context.config.embedding_batch_size})..."
        )

        if logger.isEnabledFor(logging.DEBUG):
            # Log sample text lengths
            text_lengths = [len(t) for t in texts[:5]]
            logger.debug(
                f"[Stage 2] Sample text lengths: {text_lengths}"
            )
            logger.debug(
                f"[Stage 2] Model: {self.embedding_gen.model_name if hasattr(self.embedding_gen, 'model_name') else 'unknown'}, "
                f"Dimension: {self.embedding_gen.get_dimension()}"
            )

        # Generate embeddings (synchronous operation)
        import time
        start_time = time.time()

        embeddings = self.embedding_gen.generate_batch(
            texts,
            batch_size=context.config.embedding_batch_size,
        )

        elapsed = time.time() - start_time

        # Update context
        context.embeddings = embeddings
        context.add_stat("embeddings_generated", len(embeddings))
        context.add_stat("embedding_dimension", self.embedding_gen.get_dimension())
        context.add_stat("embedding_time_seconds", round(elapsed, 2))

        logger.info(
            f"[Stage 2] Generated {len(embeddings)} embeddings "
            f"(dim={self.embedding_gen.get_dimension()}, "
            f"time={elapsed:.2f}s, "
            f"rate={len(embeddings)/elapsed:.1f} issues/sec)"
        )

        if logger.isEnabledFor(logging.DEBUG):
            # Log embedding statistics
            import numpy as np
            emb_norms = np.linalg.norm(embeddings, axis=1)
            logger.debug(
                f"[Stage 2] Embedding norms: "
                f"min={emb_norms.min():.3f}, "
                f"max={emb_norms.max():.3f}, "
                f"mean={emb_norms.mean():.3f}"
            )

        return context
