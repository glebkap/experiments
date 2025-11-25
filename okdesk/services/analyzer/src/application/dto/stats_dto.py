"""DTOs for statistics responses."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ProcessingStatsDTO:
    """Overall processing statistics."""

    total_issues: int
    processed_issues: int
    unprocessed_issues: int
    total_embeddings: int
    total_clusters: int
    clustered_issues: int

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_issues": self.total_issues,
            "processed_issues": self.processed_issues,
            "unprocessed_issues": self.unprocessed_issues,
            "total_embeddings": self.total_embeddings,
            "total_clusters": self.total_clusters,
            "clustered_issues": self.clustered_issues,
        }


@dataclass
class SourceStatsDTO:
    """Statistics for a single data source."""

    source_id: str
    source_name: str
    source_type: str
    issues_count: int
    processed_count: int
    unprocessed_count: int

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "source_id": self.source_id,
            "source_name": self.source_name,
            "source_type": self.source_type,
            "issues_count": self.issues_count,
            "processed_count": self.processed_count,
            "unprocessed_count": self.unprocessed_count,
        }


@dataclass
class TimelinePointDTO:
    """Single point in timeline statistics."""

    date: str
    issues_count: int
    processed_count: int

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "date": self.date,
            "issues_count": self.issues_count,
            "processed_count": self.processed_count,
        }


@dataclass
class ClusterStatsDTO:
    """Statistics for a single cluster."""

    cluster_id: str
    cluster_label: int
    name: Optional[str]
    size: int
    avg_distance: Optional[float]
    min_distance: Optional[float]
    max_distance: Optional[float]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "cluster_id": self.cluster_id,
            "cluster_label": self.cluster_label,
            "name": self.name,
            "size": self.size,
            "avg_distance": self.avg_distance,
            "min_distance": self.min_distance,
            "max_distance": self.max_distance,
        }
