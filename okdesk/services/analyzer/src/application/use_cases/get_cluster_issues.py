"""Use case for getting issues in a cluster."""

import logging
from typing import List, Optional
from uuid import UUID

from ...domain.repositories.cluster_repository import ClusterRepository
from ...domain.repositories.issue_repository import IssueRepository
from ..dto.issue_dto import IssueDTO

logger = logging.getLogger(__name__)


class GetClusterIssuesUseCase:
    """Use case for retrieving all issues in a cluster."""

    def __init__(
        self,
        cluster_repo: ClusterRepository,
        issue_repo: IssueRepository,
    ):
        """
        Initialize use case with dependencies.

        Args:
            cluster_repo: Cluster repository
            issue_repo: Issue repository
        """
        self.cluster_repo = cluster_repo
        self.issue_repo = issue_repo

    async def execute(
        self,
        cluster_id: UUID,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[IssueDTO]:
        """
        Get all issues in a cluster with their full information.

        Args:
            cluster_id: Cluster UUID
            limit: Maximum number of issues to return (None = all)
            offset: Number of issues to skip for pagination

        Returns:
            List of IssueDTO objects with full information

        Raises:
            ValueError: If cluster not found
        """
        logger.info(f"Fetching issues for cluster {cluster_id} (limit={limit}, offset={offset})")

        # Check if cluster exists
        cluster = await self.cluster_repo.get_cluster_by_id(cluster_id)
        if cluster is None:
            raise ValueError(f"Cluster {cluster_id} not found")

        logger.info(f"Cluster {cluster_id} found: {cluster.name or cluster.cluster_label}, size={cluster.size}")

        # Get issue IDs from cluster
        issue_ids = await self.cluster_repo.get_cluster_issues(
            cluster_id=cluster_id,
            limit=limit,
            offset=offset,
        )

        if not issue_ids:
            logger.info(f"No issues found in cluster {cluster_id}")
            return []

        logger.info(f"Found {len(issue_ids)} issue IDs in cluster")

        # Fetch full issue information
        issues = []
        for issue_id in issue_ids:
            issue = await self.issue_repo.get_by_id(issue_id)
            if issue:
                issues.append(
                    IssueDTO(
                        id=str(issue.id),
                        external_id=issue.external_id,
                        source_id=str(issue.source_id),
                        title=issue.title,
                        description=issue.description,
                        status=issue.status,
                        priority=issue.priority,
                        created_at=issue.created_at.isoformat(),
                        updated_at=issue.updated_at.isoformat() if issue.updated_at else None,
                    )
                )

        logger.info(f"Returning {len(issues)} issues from cluster {cluster_id}")
        return issues
