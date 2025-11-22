"""Embedding generation service interface."""

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class EmbeddingGenerator(ABC):
    """
    Abstract service for generating text embeddings.

    Implementation will be provided by infrastructure layer using SentenceTransformers.
    """

    @abstractmethod
    def generate_batch(
        self, texts: List[str], batch_size: int = 32, show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            numpy array of shape (len(texts), embedding_dim) with L2-normalized embeddings
        """
        pass

    @abstractmethod
    def generate_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to encode

        Returns:
            numpy array of shape (embedding_dim,) with L2-normalized embedding
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get embedding dimension.

        Returns:
            Dimension of generated embeddings
        """
        pass
