"""Repository interface for statistics operations."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List


class StatsRepository(ABC):
    """Abstract repository for statistics and aggregations."""

    @abstractmethod
    async def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get overall processing statistics.

        Returns:
            Dict with:
            - total_issues: Total number of issues in database
            - processed_issues: Number of preprocessed issues
            - unprocessed_issues: Number of issues not yet processed
            - total_embeddings: Number of embeddings in vector DB
            - total_clusters: Number of clusters
        """
        pass

    @abstractmethod
    async def get_sources_stats(self) -> List[Dict[str, Any]]:
        """
        Get statistics grouped by data source.

        Returns:
            List of dicts with:
            - source_id: UUID
            - source_name: str
            - source_type: str (okdesk, telegram)
            - issues_count: int
            - processed_count: int
            - unprocessed_count: int
        """
        pass

    @abstractmethod
    async def get_timeline_stats(
        self,
        date_from: datetime,
        date_to: datetime,
        group_by: str = "day",
    ) -> List[Dict[str, Any]]:
        """
        Get timeline statistics for issues.

        Args:
            date_from: Start date
            date_to: End date
            group_by: Grouping interval - "day", "week", "month"

        Returns:
            List of dicts with:
            - date: str (formatted date)
            - issues_count: int
            - processed_count: int
        """
        pass
