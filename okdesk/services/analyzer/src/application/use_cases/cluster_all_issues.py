"""Use case for clustering all processed issues."""

import logging
import time
from typing import List, Optional
from uuid import UUID

import numpy as np

from ...domain.repositories.cluster_repository import ClusterRepository
from ...domain.repositories.issue_repository import IssueRepository
from ...domain.services.clustering_service import ClusteringService
from ...domain.services.vector_db_service import VectorDBService
from ..dto.clustering_result_dto import ClusteringResultDTO, ClusterSummaryDTO

logger = logging.getLogger(__name__)


class ClusterAllIssuesUseCase:
    """Use case for clustering all processed issues (issue-centric)."""

    def __init__(
        self,
        vectordb: VectorDBService,
        clustering_service: ClusteringService,
        cluster_repo: ClusterRepository,
        issue_repo: IssueRepository,
    ):
        """
        Initialize use case with dependencies.

        Args:
            vectordb: Vector database service
            clustering_service: Clustering service
            cluster_repo: Cluster repository
            issue_repo: Issue repository
        """
        self.vectordb = vectordb
        self.clustering_service = clustering_service
        self.cluster_repo = cluster_repo
        self.issue_repo = issue_repo

    async def execute(
        self,
        method: str = "hdbscan",
        min_cluster_size: int = 5,
        min_samples: int = 3,
        n_clusters: Optional[int] = None,
    ) -> ClusteringResultDTO:
        """
        Cluster all processed issues using embeddings.

        Args:
            method: Clustering method ('hdbscan' or 'kmeans')
            min_cluster_size: Minimum cluster size for HDBSCAN
            min_samples: Minimum samples for HDBSCAN
            n_clusters: Number of clusters for K-means (auto if None)

        Returns:
            Clustering result with statistics

        Raises:
            ValueError: If no embeddings found or invalid method
        """
        start_time = time.time()

        logger.info(f"Starting clustering with method={method}")

        # 1. Load all embeddings from ChromaDB
        issue_ids, embeddings = await self.vectordb.get_all_embeddings()

        if len(issue_ids) == 0:
            raise ValueError("No embeddings found in vector database")

        logger.info(f"Loaded {len(issue_ids)} embeddings")

        # 2. Apply clustering algorithm
        if method == "hdbscan":
            labels, centroids = self.clustering_service.cluster_hdbscan(
                embeddings, min_cluster_size=min_cluster_size, min_samples=min_samples
            )
        elif method == "kmeans":
            labels, centroids = self.clustering_service.cluster_kmeans(
                embeddings, n_clusters=n_clusters
            )
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # 3. Compute distances to centroids
        distances = self.clustering_service.compute_distances(
            embeddings, centroids, labels
        )

        # 4. Clear existing clusters
        await self.cluster_repo.clear_all_clusters()
        logger.info("Cleared existing clusters")

        # 5. Create new clusters in database
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Exclude outliers

        cluster_map = {}  # label -> cluster_id

        for label in sorted(unique_labels):
            mask = labels == label
            size = int(mask.sum())

            cluster = await self.cluster_repo.create_cluster(
                label=f"Cluster {label}",
                centroid_embedding=centroids[label],
            )

            cluster_map[label] = cluster.id
            logger.debug(f"Created cluster {cluster.id} with {size} items")

        # 6. Assign issues to clusters (batch by cluster)
        for label, cluster_id in cluster_map.items():
            mask = labels == label
            cluster_issue_ids = [issue_ids[i] for i, m in enumerate(mask) if m]
            cluster_distances = [distances[i] for i, m in enumerate(mask) if m]

            await self.cluster_repo.assign_issues(
                cluster_id=cluster_id,
                issue_ids=cluster_issue_ids,
                distances=cluster_distances,
            )

            logger.debug(
                f"Assigned {len(cluster_issue_ids)} issues to cluster {label}"
            )

        # 7. Prepare result
        outliers_count = int((labels == -1).sum())
        duration = time.time() - start_time

        # Load created clusters for summary
        clusters = await self.cluster_repo.get_all_clusters()
        cluster_summaries = []

        for cluster in clusters:
            # Get sample issues for this cluster
            sample_titles = await self._get_sample_issue_titles(cluster.id, limit=3)

            cluster_summaries.append(
                ClusterSummaryDTO(
                    cluster_id=str(cluster.id),
                    cluster_label=int(cluster.label.split()[-1]),  # Extract number from "Cluster N"
                    name=cluster.description,
                    size=cluster.size,
                    sample_issues=sample_titles,
                )
            )

        result = ClusteringResultDTO(
            total_issues=len(issue_ids),
            num_clusters=len(unique_labels),
            outliers_count=outliers_count,
            clusters=cluster_summaries,
            duration_seconds=duration,
        )

        logger.info(
            f"Clustering completed: {len(unique_labels)} clusters, "
            f"{outliers_count} outliers in {duration:.2f}s"
        )

        return result

    async def _get_sample_issue_titles(
        self, cluster_id: UUID, limit: int = 3
    ) -> List[str]:
        """
        Get sample issue titles for a cluster.

        Args:
            cluster_id: Cluster UUID
            limit: Maximum number of titles to return

        Returns:
            List of issue titles
        """
        # This is a simplified implementation
        # In production, you'd query message_clusters -> messages -> issues
        # For now, return empty list as it requires additional repository methods
        return []
