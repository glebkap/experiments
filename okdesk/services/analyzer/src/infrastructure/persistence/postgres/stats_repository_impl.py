"""PostgreSQL implementation of StatsRepository."""

import logging
from datetime import datetime
from typing import Any, Dict, List

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ....domain.repositories.stats_repository import StatsRepository
from ....domain.services.vector_db_service import VectorDBService
from .models import (
    ClusterModel,
    IssueClusterModel,
    IssueModel,
    PreprocessedIssueModel,
    SourceModel,
)

logger = logging.getLogger(__name__)


class StatsRepositoryImpl(StatsRepository):
    """PostgreSQL implementation of StatsRepository."""

    def __init__(self, session: AsyncSession, vectordb: VectorDBService):
        """
        Initialize repository.

        Args:
            session: SQLAlchemy async session
            vectordb: Vector database service for embedding stats
        """
        self.session = session
        self.vectordb = vectordb

    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get overall processing statistics."""
        # Total issues
        total_query = select(func.count()).select_from(IssueModel)
        total_result = await self.session.execute(total_query)
        total_issues = total_result.scalar_one()

        # Processed issues (in preprocessed_issues table)
        processed_query = select(func.count()).select_from(PreprocessedIssueModel)
        processed_result = await self.session.execute(processed_query)
        processed_issues = processed_result.scalar_one()

        # Unprocessed
        unprocessed_issues = total_issues - processed_issues

        # Total embeddings in vector DB
        total_embeddings = await self.vectordb.count_total()

        # Total clusters
        clusters_query = select(func.count()).select_from(ClusterModel)
        clusters_result = await self.session.execute(clusters_query)
        total_clusters = clusters_result.scalar_one()

        # Clustered issues
        clustered_query = select(func.count(func.distinct(IssueClusterModel.issue_id))).select_from(
            IssueClusterModel
        )
        clustered_result = await self.session.execute(clustered_query)
        clustered_issues = clustered_result.scalar_one()

        stats = {
            "total_issues": total_issues,
            "processed_issues": processed_issues,
            "unprocessed_issues": unprocessed_issues,
            "total_embeddings": total_embeddings,
            "total_clusters": total_clusters,
            "clustered_issues": clustered_issues,
        }

        logger.debug(f"Processing stats: {stats}")
        return stats

    async def get_sources_stats(self) -> List[Dict[str, Any]]:
        """Get statistics grouped by data source."""
        # Get issue counts by source
        query = (
            select(
                SourceModel.id,
                SourceModel.name,
                SourceModel.type,
                func.count(IssueModel.id).label("issues_count"),
            )
            .outerjoin(IssueModel, SourceModel.id == IssueModel.source_id)
            .group_by(SourceModel.id)
            .order_by(func.count(IssueModel.id).desc())
        )

        result = await self.session.execute(query)
        rows = result.all()

        stats = []
        for row in rows:
            source_id = row[0]

            # Count processed issues for this source
            processed_query = (
                select(func.count())
                .select_from(PreprocessedIssueModel)
                .join(IssueModel, PreprocessedIssueModel.id == IssueModel.id)
                .where(IssueModel.source_id == source_id)
            )
            processed_result = await self.session.execute(processed_query)
            processed_count = processed_result.scalar_one()

            stats.append({
                "source_id": str(source_id),
                "source_name": row[1],
                "source_type": row[2],
                "issues_count": row[3],
                "processed_count": processed_count,
                "unprocessed_count": row[3] - processed_count,
            })

        logger.debug(f"Source stats: {len(stats)} sources")
        return stats

    async def get_timeline_stats(
        self,
        date_from: datetime,
        date_to: datetime,
        group_by: str = "day",
    ) -> List[Dict[str, Any]]:
        """Get timeline statistics for issues."""
        # Map group_by to PostgreSQL date_trunc interval
        interval_map = {
            "day": "day",
            "week": "week",
            "month": "month",
        }
        interval = interval_map.get(group_by, "day")

        # Query for issue counts by date
        issues_query = (
            select(
                func.date_trunc(interval, IssueModel.created_at).label("period"),
                func.count(IssueModel.id).label("issues_count"),
            )
            .where(IssueModel.created_at >= date_from)
            .where(IssueModel.created_at <= date_to)
            .group_by(text("period"))
            .order_by(text("period"))
        )

        issues_result = await self.session.execute(issues_query)
        issues_rows = issues_result.all()

        # Query for processed counts by date
        processed_query = (
            select(
                func.date_trunc(interval, PreprocessedIssueModel.processed_at).label("period"),
                func.count(PreprocessedIssueModel.id).label("processed_count"),
            )
            .where(PreprocessedIssueModel.processed_at >= date_from)
            .where(PreprocessedIssueModel.processed_at <= date_to)
            .group_by(text("period"))
            .order_by(text("period"))
        )

        processed_result = await self.session.execute(processed_query)
        processed_rows = processed_result.all()

        # Merge results
        processed_map = {row[0]: row[1] for row in processed_rows}

        stats = []
        for row in issues_rows:
            period = row[0]
            stats.append({
                "date": period.isoformat() if period else None,
                "issues_count": row[1],
                "processed_count": processed_map.get(period, 0),
            })

        logger.debug(f"Timeline stats: {len(stats)} periods from {date_from} to {date_to}")
        return stats
