"""Repository interface for Cluster entity."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np

from ..models.cluster import Cluster
from ..models.issue import Issue


class ClusterRepository(ABC):
    """Abstract repository for Cluster operations (issue-centric)."""

    @abstractmethod
    async def create_cluster(
        self, label: int, centroid_embedding: np.ndarray, name: Optional[str] = None
    ) -> Cluster:
        """
        Create new cluster.

        Args:
            label: Cluster label from clustering algorithm (int)
            centroid_embedding: Centroid vector
            name: Optional human-readable name

        Returns:
            Created Cluster domain object
        """
        pass

    @abstractmethod
    async def assign_issues(
        self, cluster_id: UUID, issue_ids: List[UUID], distances: List[float]
    ) -> None:
        """
        Assign issues to a cluster with distances to centroid.

        Uses batch insert for performance.

        Args:
            cluster_id: Target cluster UUID
            issue_ids: List of issue UUIDs to assign
            distances: List of cosine distances to centroid (0-1)
        """
        pass

    @abstractmethod
    async def get_all_clusters(self) -> List[Cluster]:
        """
        Get all clusters.

        Returns:
            List of all clusters, ordered by size DESC
        """
        pass

    @abstractmethod
    async def get_cluster_by_id(self, cluster_id: UUID) -> Optional[Cluster]:
        """
        Get cluster by ID.

        Args:
            cluster_id: Cluster UUID

        Returns:
            Cluster if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_cluster_assignments(self) -> Dict[UUID, UUID]:
        """
        Get all issue-to-cluster assignments.

        Returns:
            Dictionary mapping issue_id -> cluster_id
        """
        pass

    @abstractmethod
    async def clear_all_clusters(self) -> None:
        """
        Delete all clusters and their assignments.

        Used before re-clustering.
        """
        pass

    @abstractmethod
    async def count_clusters(self) -> int:
        """
        Count total number of clusters.

        Returns:
            Number of clusters
        """
        pass

    @abstractmethod
    async def get_cluster_issues(
        self, cluster_id: UUID, limit: Optional[int] = None, offset: int = 0
    ) -> List[UUID]:
        """
        Get issue IDs assigned to a cluster.

        Args:
            cluster_id: Cluster UUID
            limit: Maximum number of issues to return (None = all)
            offset: Number of issues to skip

        Returns:
            List of issue UUIDs in the cluster
        """
        pass

    # ==================== Extension methods for 04-analyzer-extensions ====================

    @abstractmethod
    async def get_cluster_with_issues(
        self,
        cluster_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> Optional[Tuple[Cluster, List[Issue], int]]:
        """
        Get cluster with its issues (full Issue objects).

        Issues are sorted by distance_to_centroid ASC (closest first).

        Args:
            cluster_id: Cluster UUID
            limit: Maximum number of issues to return
            offset: Number of issues to skip

        Returns:
            Tuple of (Cluster, List[Issue], total_count) if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_cluster_stats(self) -> List[Dict[str, Any]]:
        """
        Get statistics for all clusters.

        Returns:
            List of dicts with cluster stats:
            - cluster_id: UUID
            - cluster_label: int
            - name: Optional[str]
            - size: int
            - avg_distance: float
            - min_distance: float
            - max_distance: float
        """
        pass

    @abstractmethod
    async def get_issue_cluster(self, issue_id: UUID) -> Optional[Tuple[UUID, int, float]]:
        """
        Get cluster info for a specific issue.

        Args:
            issue_id: Issue UUID

        Returns:
            Tuple of (cluster_id, cluster_label, distance_to_centroid) or None
        """
        pass
