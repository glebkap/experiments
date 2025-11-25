"""PostgreSQL implementation of ClusterRepository."""

import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import numpy as np
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ....domain.models.cluster import Cluster
from ....domain.models.issue import Issue
from ....domain.repositories.cluster_repository import ClusterRepository
from .models import ClusterModel, IssueClusterModel, IssueModel

logger = logging.getLogger(__name__)


class ClusterRepositoryImpl(ClusterRepository):
    """PostgreSQL implementation of ClusterRepository."""

    def __init__(self, session: AsyncSession):
        """
        Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        self.session = session

    async def create_cluster(
        self, label: int, centroid_embedding: np.ndarray, name: Optional[str] = None
    ) -> Cluster:
        """
        Create a new cluster.

        Args:
            label: Cluster label from clustering algorithm (int)
            centroid_embedding: Centroid vector
            name: Optional human-readable name

        Returns:
            Created Cluster domain object
        """
        logger.debug(f"Creating cluster with label {label}")

        cluster_id = uuid4()

        # Convert numpy array to list for PostgreSQL ARRAY type
        centroid_list = centroid_embedding.tolist()

        model = ClusterModel(
            id=cluster_id,
            cluster_label=label,
            name=name,
            description=None,
            centroid_embedding=centroid_list,
            size=0,
        )

        self.session.add(model)
        await self.session.commit()
        await self.session.refresh(model)

        logger.info(f"Created cluster {cluster_id} with label {label}")

        return self._to_domain(model)

    async def assign_issues(
        self, cluster_id: UUID, issue_ids: List[UUID], distances: List[float]
    ) -> None:
        """
        Assign issues to a cluster with distances.

        Args:
            cluster_id: Cluster UUID
            issue_ids: List of issue UUIDs
            distances: List of distances from cluster centroid
        """
        if not issue_ids:
            logger.debug("No issues to assign")
            return

        if len(issue_ids) != len(distances):
            raise ValueError(
                f"Issue IDs count ({len(issue_ids)}) doesn't match distances count ({len(distances)})"
            )

        logger.debug(f"Assigning {len(issue_ids)} issues to cluster {cluster_id}")

        # Delete existing assignments for these issues
        delete_stmt = delete(IssueClusterModel).where(
            IssueClusterModel.issue_id.in_(issue_ids)
        )
        await self.session.execute(delete_stmt)

        # Create new assignments
        assignments = [
            IssueClusterModel(
                issue_id=issue_id,
                cluster_id=cluster_id,
                distance_to_centroid=distance,
            )
            for issue_id, distance in zip(issue_ids, distances)
        ]

        self.session.add_all(assignments)

        # Update cluster size
        update_stmt = (
            update(ClusterModel)
            .where(ClusterModel.id == cluster_id)
            .values(size=len(issue_ids))
        )
        await self.session.execute(update_stmt)

        await self.session.commit()

        logger.info(
            f"Successfully assigned {len(issue_ids)} issues to cluster {cluster_id}"
        )

    async def get_cluster_assignments(self) -> Dict[UUID, UUID]:
        """
        Get all issue-to-cluster assignments.

        Returns:
            Dictionary mapping issue_id -> cluster_id
        """
        logger.debug("Fetching all cluster assignments")

        query = select(
            IssueClusterModel.issue_id, IssueClusterModel.cluster_id
        )
        result = await self.session.execute(query)
        rows = result.all()

        assignments = {row[0]: row[1] for row in rows}

        logger.debug(f"Found {len(assignments)} cluster assignments")

        return assignments

    async def clear_all_clusters(self) -> None:
        """
        Delete all clusters and their assignments.

        This is useful for re-clustering from scratch.
        """
        logger.warning("Clearing all clusters and assignments")

        # Delete all issue-cluster assignments first (foreign key constraint)
        await self.session.execute(delete(IssueClusterModel))

        # Delete all clusters
        await self.session.execute(delete(ClusterModel))

        await self.session.commit()

        logger.info("Successfully cleared all clusters")

    async def get_all_clusters(self) -> List[Cluster]:
        """
        Get all clusters.

        Returns:
            List of all Cluster domain objects
        """
        logger.debug("Fetching all clusters")

        query = select(ClusterModel).order_by(ClusterModel.size.desc())
        result = await self.session.execute(query)
        models = result.scalars().all()

        logger.debug(f"Found {len(models)} clusters")

        return [self._to_domain(model) for model in models]

    async def get_cluster_by_id(self, cluster_id: UUID) -> Optional[Cluster]:
        """
        Get cluster by ID.

        Args:
            cluster_id: Cluster UUID

        Returns:
            Cluster domain object or None
        """
        query = select(ClusterModel).where(ClusterModel.id == cluster_id)
        result = await self.session.execute(query)
        model = result.scalar_one_or_none()

        if model is None:
            logger.debug(f"Cluster {cluster_id} not found")
            return None

        return self._to_domain(model)

    async def count_clusters(self) -> int:
        """
        Count total number of clusters.

        Returns:
            Total cluster count
        """
        query = select(func.count()).select_from(ClusterModel)
        result = await self.session.execute(query)
        count = result.scalar_one()

        logger.debug(f"Total clusters: {count}")
        return count

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
        logger.debug(f"Fetching issues for cluster {cluster_id} (limit={limit}, offset={offset})")

        query = (
            select(IssueClusterModel.issue_id)
            .where(IssueClusterModel.cluster_id == cluster_id)
            .order_by(IssueClusterModel.distance_to_centroid)  # Closest first
            .offset(offset)
        )

        if limit is not None:
            query = query.limit(limit)

        result = await self.session.execute(query)
        issue_ids = [row[0] for row in result.all()]

        logger.debug(f"Found {len(issue_ids)} issues in cluster {cluster_id}")
        return issue_ids

    def _to_domain(self, model: ClusterModel) -> Cluster:
        """
        Convert SQLAlchemy model to domain object.

        Args:
            model: ClusterModel instance

        Returns:
            Cluster domain object
        """
        # Convert PostgreSQL ARRAY back to numpy array
        centroid = np.array(model.centroid_embedding, dtype=np.float32)

        return Cluster(
            id=model.id,
            cluster_label=model.cluster_label,
            name=model.name,
            description=model.description,
            centroid_embedding=centroid,
            size=model.size,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    def _issue_to_domain(self, model: IssueModel) -> Issue:
        """Convert IssueModel to domain Issue."""
        return Issue(
            id=model.id,
            external_id=model.external_id,
            source_id=model.source_id,
            title=model.title,
            description=model.description,
            status=model.status,
            priority=model.priority,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    # ==================== Extension methods for 04-analyzer-extensions ====================

    async def get_cluster_with_issues(
        self,
        cluster_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> Optional[Tuple[Cluster, List[Issue], int]]:
        """Get cluster with its issues (full Issue objects)."""
        # Get cluster
        cluster_query = select(ClusterModel).where(ClusterModel.id == cluster_id)
        cluster_result = await self.session.execute(cluster_query)
        cluster_model = cluster_result.scalar_one_or_none()

        if cluster_model is None:
            logger.debug(f"Cluster {cluster_id} not found")
            return None

        cluster = self._to_domain(cluster_model)

        # Count total issues in cluster
        count_query = (
            select(func.count())
            .select_from(IssueClusterModel)
            .where(IssueClusterModel.cluster_id == cluster_id)
        )
        count_result = await self.session.execute(count_query)
        total_count = count_result.scalar_one()

        # Get issues with pagination, ordered by distance to centroid (closest first)
        issues_query = (
            select(IssueModel)
            .join(IssueClusterModel, IssueModel.id == IssueClusterModel.issue_id)
            .where(IssueClusterModel.cluster_id == cluster_id)
            .order_by(IssueClusterModel.distance_to_centroid)
            .limit(limit)
            .offset(offset)
        )
        issues_result = await self.session.execute(issues_query)
        issue_models = issues_result.scalars().all()

        issues = [self._issue_to_domain(m) for m in issue_models]

        logger.debug(
            f"Found cluster {cluster_id} with {len(issues)} issues (total: {total_count})"
        )
        return (cluster, issues, total_count)

    async def get_cluster_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all clusters."""
        query = (
            select(
                ClusterModel.id,
                ClusterModel.cluster_label,
                ClusterModel.name,
                ClusterModel.size,
                func.avg(IssueClusterModel.distance_to_centroid).label("avg_distance"),
                func.min(IssueClusterModel.distance_to_centroid).label("min_distance"),
                func.max(IssueClusterModel.distance_to_centroid).label("max_distance"),
            )
            .outerjoin(IssueClusterModel, ClusterModel.id == IssueClusterModel.cluster_id)
            .group_by(ClusterModel.id)
            .order_by(ClusterModel.size.desc())
        )

        result = await self.session.execute(query)
        rows = result.all()

        stats = []
        for row in rows:
            stats.append({
                "cluster_id": str(row[0]),
                "cluster_label": row[1],
                "name": row[2],
                "size": row[3],
                "avg_distance": float(row[4]) if row[4] is not None else None,
                "min_distance": float(row[5]) if row[5] is not None else None,
                "max_distance": float(row[6]) if row[6] is not None else None,
            })

        logger.debug(f"Computed stats for {len(stats)} clusters")
        return stats

    async def get_issue_cluster(self, issue_id: UUID) -> Optional[Tuple[UUID, int, float]]:
        """Get cluster info for a specific issue."""
        query = (
            select(
                IssueClusterModel.cluster_id,
                ClusterModel.cluster_label,
                IssueClusterModel.distance_to_centroid,
            )
            .join(ClusterModel, IssueClusterModel.cluster_id == ClusterModel.id)
            .where(IssueClusterModel.issue_id == issue_id)
        )

        result = await self.session.execute(query)
        row = result.one_or_none()

        if row is None:
            logger.debug(f"No cluster assignment found for issue {issue_id}")
            return None

        return (row[0], row[1], row[2])
