"""PostgreSQL implementation of ClusterRepository."""

import logging
from typing import Dict, List, Optional
from uuid import UUID, uuid4

import numpy as np
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ....domain.models.cluster import Cluster
from ....domain.repositories.cluster_repository import ClusterRepository
from .models import ClusterModel, IssueClusterModel

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
