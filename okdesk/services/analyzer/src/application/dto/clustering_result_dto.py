"""DTOs for clustering results."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ClusterSummaryDTO:
    """Summary of a single cluster."""

    cluster_id: str
    cluster_label: int
    name: Optional[str]
    size: int
    sample_issues: List[str]  # Sample issue titles


@dataclass
class ClusteringResultDTO:
    """Result of clustering operation."""

    total_issues: int
    num_clusters: int
    outliers_count: int
    clusters: List[ClusterSummaryDTO]
    duration_seconds: float
    method: str  # hdbscan or kmeans

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "total_issues": self.total_issues,
            "num_clusters": self.num_clusters,
            "outliers_count": self.outliers_count,
            "clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "cluster_label": c.cluster_label,
                    "name": c.name,
                    "size": c.size,
                    "sample_issues": c.sample_issues,
                }
                for c in self.clusters
            ],
            "duration_seconds": round(self.duration_seconds, 2),
            "method": self.method,
        }
