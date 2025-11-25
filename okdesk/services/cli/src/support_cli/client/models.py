"""Pydantic models for API responses."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# Import models
class ImportResponse(BaseModel):
    """Response from import endpoint."""

    import_id: str
    status: Literal["in_progress", "completed", "failed"]
    message: str


class ImportStats(BaseModel):
    """Import statistics."""

    total_issues: int = 0
    new_issues: int = 0
    updated_issues: int = 0
    total_messages: int = 0
    new_messages: int = 0
    skipped_duplicates: int = 0


class ImportStatus(BaseModel):
    """Import status details."""

    id: str
    source_id: str | None = None
    filename: str | None = None
    started_at: datetime
    completed_at: datetime | None = None
    status: Literal["in_progress", "completed", "failed"]
    stats: ImportStats | None = None
    error_message: str | None = None


class ImportHistory(BaseModel):
    """Import history item."""

    id: str
    filename: str | None = None
    started_at: datetime
    completed_at: datetime | None = None
    status: Literal["in_progress", "completed", "failed"]
    stats: ImportStats | None = None


# Pipeline models
class PipelineStats(BaseModel):
    """Pipeline processing statistics."""

    stage_1_preprocessed: int = 0
    stage_2_embeddings: int = 0
    stage_3_vector_db: int = 0
    stage_4_completed: int = 0
    failed: int = 0
    total_duration_seconds: float | None = None


class PipelineResponse(BaseModel):
    """Response from pipeline process endpoint."""

    processed_count: int
    duration_seconds: float
    stats: PipelineStats


class PipelineStatus(BaseModel):
    """Pipeline status information."""

    total_issues: int
    processed_issues: int
    pending_issues: int
    success_rate: float | None = None
    last_processed_at: datetime | None = None


# Clustering models
class ClusterInfo(BaseModel):
    """Cluster information."""

    id: str
    cluster_label: int
    name: str | None = None
    description: str | None = None
    size: int
    created_at: datetime


class ClusteringInfo(BaseModel):
    """Clustering information."""

    total_clusters: int
    total_issues_clustered: int
    noise_points: int
    clusters: list[ClusterInfo] = Field(default_factory=list)


class ClusterDetails(BaseModel):
    """Detailed cluster information with issues."""

    cluster: ClusterInfo
    issues: list["IssueListItem"]


# Issue models
class IssueListItem(BaseModel):
    """Issue list item (summary)."""

    id: str
    external_id: str
    title: str | None = None
    status: Literal["opened", "wait", "completed", "closed"] | None = None
    priority: int | None = None
    created_at: datetime | None = None
    cluster_id: str | None = None
    similarity: float | None = None  # For search results


class MessageItem(BaseModel):
    """Message item."""

    id: str
    external_id: str
    author_name: str | None = None
    author_type: Literal["employee", "contact", "user"] | None = None
    content: str
    is_public: bool = True
    published_at: datetime | None = None


class IssueDetails(BaseModel):
    """Detailed issue information."""

    id: str
    external_id: str
    title: str | None = None
    description: str | None = None
    status: Literal["opened", "wait", "completed", "closed"] | None = None
    priority: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None
    messages: list[MessageItem] = Field(default_factory=list)
    cluster_id: str | None = None


# Search models
class SearchResult(BaseModel):
    """Semantic search result."""

    issue_id: str
    similarity: float
    text_snippet: str
    title: str | None = None
    created_at: datetime | None = None


# Stats models
class ProcessingStats(BaseModel):
    """Processing statistics."""

    total_issues: int
    processed: int
    pending: int
    success_rate: float
    average_processing_time: float | None = None


class ClusterStats(BaseModel):
    """Cluster statistics."""

    total_clusters: int
    total_issues_clustered: int
    noise_points: int
    largest_cluster_size: int
    average_cluster_size: float
    top_clusters: list[ClusterInfo] = Field(default_factory=list)


class SourceStats(BaseModel):
    """Source statistics."""

    source_id: str
    source_name: str
    source_type: Literal["okdesk", "telegram"]
    total_issues: int
    processed: int
    pending: int


class TimelinePoint(BaseModel):
    """Timeline data point."""

    period: str  # Date string or period label
    count: int


class TimelineStats(BaseModel):
    """Timeline statistics."""

    granularity: Literal["day", "week", "month"]
    from_date: datetime
    to_date: datetime
    data_points: list[TimelinePoint] = Field(default_factory=list)


class StatsResponse(BaseModel):
    """Generic stats response wrapper."""

    data: (
        ProcessingStats
        | ClusterStats
        | list[SourceStats]
        | TimelineStats
        | dict[str, Any]
    )


# Export models
class ExportResponse(BaseModel):
    """Export response."""

    filename: str
    format: Literal["csv", "json"]
    total_records: int
    file_size_bytes: int | None = None
    download_url: str | None = None
