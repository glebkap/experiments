"""Vector database service interface."""

from abc import ABC, abstractmethod
from typing import List, Tuple
from uuid import UUID

import numpy as np


class VectorDBService(ABC):
    """
    Abstract service for vector database operations (ChromaDB).

    Implementation will be provided by infrastructure layer.
    """

    @abstractmethod
    def save_embeddings(
        self, issue_ids: List[UUID], embeddings: np.ndarray, documents: List[str]
    ) -> None:
        """
        Save embeddings to vector database.

        Args:
            issue_ids: List of issue UUIDs
            embeddings: numpy array of shape (n_issues, embedding_dim)
            documents: List of preprocessed texts (for display in search results)
        """
        pass

    @abstractmethod
    def search_similar(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> List[Tuple[UUID, float]]:
        """
        Search for similar issues using embedding.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of (issue_id, similarity_score) tuples, ordered by similarity DESC
        """
        pass

    @abstractmethod
    def search_similar_by_id(
        self, issue_id: UUID, top_k: int = 10
    ) -> List[Tuple[UUID, float]]:
        """
        Search for similar issues using issue ID.

        Args:
            issue_id: Issue UUID to find similar to
            top_k: Number of results to return

        Returns:
            List of (issue_id, similarity_score) tuples, ordered by similarity DESC
        """
        pass

    @abstractmethod
    def get_all_embeddings(self) -> Tuple[List[UUID], np.ndarray]:
        """
        Get all embeddings from vector database.

        Used for clustering.

        Returns:
            Tuple of (issue_ids, embeddings_matrix)
            - issue_ids: List of issue UUIDs
            - embeddings_matrix: numpy array of shape (n_issues, embedding_dim)
        """
        pass

    @abstractmethod
    def delete_embedding(self, issue_id: UUID) -> None:
        """
        Delete embedding for an issue.

        Used when reprocessing an issue.

        Args:
            issue_id: Issue UUID to delete
        """
        pass

    @abstractmethod
    def count_total(self) -> int:
        """
        Count total number of embeddings in database.

        Returns:
            Number of stored embeddings
        """
        pass

    @abstractmethod
    def exists(self, issue_id: UUID) -> bool:
        """
        Check if embedding exists for an issue.

        Args:
            issue_id: Issue UUID to check

        Returns:
            True if embedding exists, False otherwise
        """
        pass

    @abstractmethod
    def clear_all(self) -> int:
        """
        Delete all embeddings from vector database.

        WARNING: This will delete the entire collection and recreate it!
        Used when changing embedding model or dimension.

        Returns:
            Number of deleted embeddings
        """
        pass
