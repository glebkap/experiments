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

        logger.info("=" * 70)
        logger.info(f"[Clustering] Starting clustering with method={method}")
        logger.info("=" * 70)

        # 1. Load all embeddings from ChromaDB
        logger.info("[Step 1/6] Loading embeddings from ChromaDB...")
        issue_ids, embeddings = await self.vectordb.get_all_embeddings()

        if len(issue_ids) == 0:
            raise ValueError("No embeddings found in vector database")

        logger.info(f"[Step 1/6] ✓ Loaded {len(issue_ids)} embeddings (shape: {embeddings.shape})")

        # 2. Apply clustering algorithm
        logger.info(f"[Step 2/6] Running {method.upper()} clustering...")
        if method == "hdbscan":
            logger.info(f"[Step 2/6] Parameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
            clustering_start = time.time()
            labels, centroids = self.clustering_service.cluster_hdbscan(
                embeddings, min_cluster_size=min_cluster_size, min_samples=min_samples
            )
            clustering_duration = time.time() - clustering_start
        elif method == "kmeans":
            logger.info(f"[Step 2/6] Parameters: n_clusters={n_clusters or 'auto'}")
            clustering_start = time.time()
            labels, centroids = self.clustering_service.cluster_kmeans(
                embeddings, n_clusters=n_clusters
            )
            clustering_duration = time.time() - clustering_start
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        logger.info(f"[Step 2/6] ✓ Clustering completed in {clustering_duration:.2f}s")

        # 3. Compute distances to centroids
        logger.info("[Step 3/6] Computing distances to cluster centroids...")
        distances = self.clustering_service.compute_distances(
            embeddings, centroids, labels
        )
        logger.info(f"[Step 3/6] ✓ Computed {len(distances)} distances")

        # 4. Clear existing clusters
        logger.info("[Step 4/6] Clearing existing clusters from database...")
        deleted_count = await self.cluster_repo.clear_all_clusters()
        logger.info(f"[Step 4/6] ✓ Cleared {deleted_count} existing clusters")

        # 5. Create new clusters in database
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Exclude outliers

        outliers_count = int((labels == -1).sum())
        logger.info(f"[Step 5/6] Creating {len(unique_labels)} clusters in database...")
        logger.info(f"[Step 5/6] Found {outliers_count} outliers (not assigned to any cluster)")

        cluster_map = {}  # label -> cluster_id

        for idx, label in enumerate(sorted(unique_labels), 1):
            mask = labels == label
            size = int(mask.sum())

            cluster = await self.cluster_repo.create_cluster(
                label=int(label),
                centroid_embedding=centroids[label],
                name=f"Cluster {label}",
            )

            cluster_map[label] = cluster.id
            logger.info(f"[Step 5/6] Created cluster {idx}/{len(unique_labels)}: label={label}, size={size}, id={cluster.id}")

        # 6. Assign issues to clusters (batch by cluster)
        logger.info(f"[Step 6/6] Assigning {len(issue_ids) - outliers_count} issues to clusters...")
        total_assigned = 0

        for idx, (label, cluster_id) in enumerate(cluster_map.items(), 1):
            mask = labels == label
            cluster_issue_ids = [issue_ids[i] for i, m in enumerate(mask) if m]
            cluster_distances = [distances[i] for i, m in enumerate(mask) if m]

            await self.cluster_repo.assign_issues(
                cluster_id=cluster_id,
                issue_ids=cluster_issue_ids,
                distances=cluster_distances,
            )

            total_assigned += len(cluster_issue_ids)
            logger.info(
                f"[Step 6/6] Assigned {len(cluster_issue_ids)} issues to cluster {label} "
                f"({idx}/{len(cluster_map)}, total: {total_assigned}/{len(issue_ids) - outliers_count})"
            )

        # 7. Prepare result
        logger.info("[Step 6/6] ✓ All issues assigned to clusters")

        duration = time.time() - start_time

        # Load created clusters for summary
        logger.info("Loading cluster summaries...")
        clusters = await self.cluster_repo.get_all_clusters()
        cluster_summaries = []

        for cluster in clusters:
            # Get sample issues for this cluster
            sample_titles = await self._get_sample_issue_titles(cluster.id, limit=3)

            cluster_summaries.append(
                ClusterSummaryDTO(
                    cluster_id=str(cluster.id),
                    cluster_label=cluster.cluster_label,
                    name=cluster.name or f"Cluster {cluster.cluster_label}",
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
            method=method,
        )

        logger.info("=" * 70)
        logger.info("[Clustering] ✓ Completed successfully!")
        logger.info(f"  • Total issues: {len(issue_ids)}")
        logger.info(f"  • Clusters created: {len(unique_labels)}")
        logger.info(f"  • Outliers: {outliers_count} ({outliers_count/len(issue_ids)*100:.1f}%)")
        logger.info(f"  • Average cluster size: {(len(issue_ids)-outliers_count)/len(unique_labels):.1f}")
        logger.info(f"  • Clustering algorithm time: {clustering_duration:.2f}s")
        logger.info(f"  • Total duration: {duration:.2f}s")
        logger.info("=" * 70)

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
