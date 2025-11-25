"""Pydantic schemas for API request/response validation."""

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from ...application.dto.clustering_result_dto import ClusterSummaryDTO
from ...application.dto.similar_issue_dto import SimilarIssueDTO


# ==================== Pipeline Schemas ====================
class PipelineRequest(BaseModel):
    """Request schema for pipeline processing."""

    batch_size: int = Field(default=100, ge=1, le=1000, description="Batch size")
    device: str = Field(default="cpu", description="Device for embeddings (cpu/cuda/mps/auto)")


class PipelineResponse(BaseModel):
    """Response schema for pipeline processing."""

    success: bool = Field(description="Whether processing succeeded")
    processed_count: int = Field(description="Number of issues processed")
    duration_seconds: float = Field(description="Total processing duration")
    stats: Dict[str, Any] = Field(description="Processing statistics")
    errors: List[str] = Field(default_factory=list, description="Error messages")


# ==================== Status Schemas ====================
class ServiceStatusResponse(BaseModel):
    """Response schema for service status."""

    total_issues: int = Field(description="Total issues in database")
    unprocessed_issues: int = Field(description="Number of unprocessed issues")
    processed_issues: int = Field(description="Number of processed issues")
    total_embeddings: int = Field(description="Total embeddings in ChromaDB")
    total_clusters: int = Field(description="Total clusters")


# ==================== Cluster Info Schema ====================
class ClusterInfoResponse(BaseModel):
    """Response schema for cluster information."""

    id: str = Field(description="Cluster UUID")
    cluster_label: int = Field(description="Cluster label (numeric)")
    name: Optional[str] = Field(None, description="Cluster name")
    size: int = Field(description="Number of issues in cluster")
    description: Optional[str] = Field(None, description="Cluster description")


# ==================== Processing Manager Schemas ====================
class ProcessingStatusResponse(BaseModel):
    """Response schema for processing status."""

    state: str = Field(description="Current state (stopped/running/paused)")
    batch_size: int = Field(description="Batch size for processing")
    poll_interval_seconds: float = Field(description="Poll interval in seconds")
    device: str = Field(description="Device for embeddings (cpu/cuda/mps/auto)")
    total_processed: int = Field(description="Total issues processed")
    total_batches: int = Field(description="Total batches processed")
    started_at: Optional[str] = Field(None, description="Processing start time (ISO)")
    last_batch_at: Optional[str] = Field(None, description="Last batch time (ISO)")
    uptime_seconds: Optional[float] = Field(None, description="Uptime in seconds")


# ==================== Clustering Schemas (extended) ====================
class ClusteringRequest(BaseModel):
    """Request schema for clustering."""

    method: str = Field(
        default="hdbscan", description="Clustering method (hdbscan or kmeans)"
    )
    min_cluster_size: int = Field(
        default=5, ge=2, description="Minimum cluster size (HDBSCAN)"
    )
    min_samples: int = Field(
        default=3, ge=1, description="Minimum samples (HDBSCAN)"
    )
    n_clusters: Optional[int] = Field(
        None, ge=2, le=100, description="Number of clusters (K-means)"
    )


class ClusteringResponse(BaseModel):
    """Response schema for clustering results."""

    success: bool = Field(description="Whether clustering succeeded")
    total_issues: int = Field(description="Total issues clustered")
    num_clusters: int = Field(description="Number of clusters created")
    outliers_count: int = Field(description="Number of outliers")
    duration_seconds: float = Field(description="Clustering duration")
    clusters: List[ClusterSummaryDTO] = Field(description="Cluster information")


# ==================== Search Schemas (extended) ====================
class SearchRequest(BaseModel):
    """Request schema for similarity search."""

    issue_id: Optional[UUID] = Field(None, description="Issue ID to find similar")
    query: Optional[str] = Field(None, description="Free-form search query")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")
    min_similarity: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum similarity threshold"
    )


class SearchResponse(BaseModel):
    """Response schema for similarity search."""

    query: str = Field(description="Search query")
    results: List[SimilarIssueDTO] = Field(description="Similar issues found")
    total_results: int = Field(description="Total number of results")


# ==================== Reprocess Schema ====================
class ReprocessRequest(BaseModel):
    """Request schema for reprocessing an issue."""

    issue_id: UUID = Field(description="Issue UUID to reprocess")


# ==================== Issues List Schemas (04-analyzer-extensions) ====================
class IssueFiltersSchema(BaseModel):
    """Query parameters for filtering issues."""

    status: Optional[str] = Field(None, description="Filter by status (opened/wait/completed/closed)")
    source_id: Optional[UUID] = Field(None, description="Filter by source UUID")
    priority: Optional[int] = Field(None, ge=1, le=4, description="Filter by priority (1-4)")
    date_from: Optional[str] = Field(None, description="Filter by created_at >= date (ISO format)")
    date_to: Optional[str] = Field(None, description="Filter by created_at <= date (ISO format)")
    limit: int = Field(default=50, ge=1, le=100, description="Maximum results to return")
    offset: int = Field(default=0, ge=0, description="Number of results to skip")


class MessageSchema(BaseModel):
    """Schema for a message in an issue."""

    id: str = Field(description="Message UUID")
    external_id: str = Field(description="External message ID")
    author_name: Optional[str] = Field(None, description="Author name")
    author_type: str = Field(description="Author type (employee/contact/user)")
    content: str = Field(description="Message content")
    is_public: bool = Field(description="Is message public")
    published_at: Optional[str] = Field(None, description="Publication time (ISO)")


class IssueListItemSchema(BaseModel):
    """Schema for a single issue in a list."""

    id: str = Field(description="Issue UUID")
    external_id: str = Field(description="External issue ID")
    title: Optional[str] = Field(None, description="Issue title")
    status: str = Field(description="Issue status")
    priority: Optional[int] = Field(None, description="Issue priority (1-4)")
    created_at: str = Field(description="Creation time (ISO)")
    source_name: Optional[str] = Field(None, description="Source name")
    source_type: Optional[str] = Field(None, description="Source type")


class IssueListResponseSchema(BaseModel):
    """Response schema for paginated issue list."""

    items: List[IssueListItemSchema] = Field(description="List of issues")
    total: int = Field(description="Total matching issues")
    limit: int = Field(description="Limit used")
    offset: int = Field(description="Offset used")


class IssueDetailSchema(BaseModel):
    """Detailed issue schema with messages."""

    id: str = Field(description="Issue UUID")
    external_id: str = Field(description="External issue ID")
    title: Optional[str] = Field(None, description="Issue title")
    description: Optional[str] = Field(None, description="Issue description")
    status: str = Field(description="Issue status")
    priority: Optional[int] = Field(None, description="Issue priority (1-4)")
    created_at: str = Field(description="Creation time (ISO)")
    updated_at: Optional[str] = Field(None, description="Last update time (ISO)")
    completed_at: Optional[str] = Field(None, description="Completion time (ISO)")
    source_name: Optional[str] = Field(None, description="Source name")
    source_type: Optional[str] = Field(None, description="Source type")
    messages: List[MessageSchema] = Field(default_factory=list, description="Issue messages")
    cluster_id: Optional[str] = Field(None, description="Cluster UUID if assigned")
    cluster_label: Optional[int] = Field(None, description="Cluster label")
    distance_to_centroid: Optional[float] = Field(None, description="Distance to cluster centroid")


# ==================== Full-text Search Schemas ====================
class FulltextSearchSchema(BaseModel):
    """Query parameters for full-text search."""

    q: str = Field(min_length=1, description="Search query text")
    limit: int = Field(default=50, ge=1, le=100, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Results to skip")


class FulltextSearchResponseSchema(BaseModel):
    """Response schema for full-text search."""

    query: str = Field(description="Search query used")
    items: List[IssueListItemSchema] = Field(description="Matching issues")
    total: int = Field(description="Total matching issues")
    limit: int = Field(description="Limit used")
    offset: int = Field(description="Offset used")


# ==================== Stats Schemas (04-analyzer-extensions) ====================
class ProcessingStatsSchema(BaseModel):
    """Processing statistics response."""

    total_issues: int = Field(description="Total issues in database")
    processed_issues: int = Field(description="Processed issues count")
    unprocessed_issues: int = Field(description="Unprocessed issues count")
    total_embeddings: int = Field(description="Total embeddings in vector DB")
    total_clusters: int = Field(description="Total clusters count")
    clustered_issues: int = Field(description="Issues assigned to clusters")


class SourceStatsSchema(BaseModel):
    """Statistics for a single source."""

    source_id: str = Field(description="Source UUID")
    source_name: str = Field(description="Source name")
    source_type: str = Field(description="Source type (okdesk/telegram)")
    issues_count: int = Field(description="Total issues from source")
    processed_count: int = Field(description="Processed issues count")
    unprocessed_count: int = Field(description="Unprocessed issues count")


class SourcesStatsResponseSchema(BaseModel):
    """Response schema for sources statistics."""

    sources: List[SourceStatsSchema] = Field(description="Statistics by source")


class ClusterStatsSchema(BaseModel):
    """Statistics for a single cluster."""

    cluster_id: str = Field(description="Cluster UUID")
    cluster_label: int = Field(description="Cluster numeric label")
    name: Optional[str] = Field(None, description="Cluster name")
    size: int = Field(description="Number of issues")
    avg_distance: Optional[float] = Field(None, description="Average distance to centroid")
    min_distance: Optional[float] = Field(None, description="Minimum distance to centroid")
    max_distance: Optional[float] = Field(None, description="Maximum distance to centroid")


class ClustersStatsResponseSchema(BaseModel):
    """Response schema for clusters statistics."""

    clusters: List[ClusterStatsSchema] = Field(description="Statistics by cluster")
    total_clusters: int = Field(description="Total number of clusters")


class TimelinePointSchema(BaseModel):
    """Single point in timeline statistics."""

    date: str = Field(description="Date (ISO format)")
    issues_count: int = Field(description="Issues created on this date")
    processed_count: int = Field(description="Issues processed on this date")


class TimelineStatsRequestSchema(BaseModel):
    """Request parameters for timeline statistics."""

    date_from: str = Field(description="Start date (ISO format)")
    date_to: str = Field(description="End date (ISO format)")
    group_by: str = Field(default="day", description="Grouping: day, week, month")


class TimelineStatsResponseSchema(BaseModel):
    """Response schema for timeline statistics."""

    date_from: str = Field(description="Start date")
    date_to: str = Field(description="End date")
    group_by: str = Field(description="Grouping used")
    points: List[TimelinePointSchema] = Field(description="Timeline data points")


# ==================== Export Schemas ====================
class ExportFiltersSchema(BaseModel):
    """Filters for export."""

    status: Optional[str] = Field(None, description="Filter by status")
    source_id: Optional[UUID] = Field(None, description="Filter by source")
    priority: Optional[int] = Field(None, ge=1, le=4, description="Filter by priority")
    date_from: Optional[str] = Field(None, description="Filter from date")
    date_to: Optional[str] = Field(None, description="Filter to date")


class ExportRequestSchema(BaseModel):
    """Request schema for data export."""

    format: str = Field(default="csv", pattern="^(csv|json)$", description="Export format")
    filters: Optional[ExportFiltersSchema] = Field(None, description="Optional filters")


# ==================== Cluster Issues Schema ====================
class ClusterIssuesResponseSchema(BaseModel):
    """Response schema for cluster issues."""

    cluster_id: str = Field(description="Cluster UUID")
    cluster_label: int = Field(description="Cluster label")
    cluster_name: Optional[str] = Field(None, description="Cluster name")
    items: List[IssueListItemSchema] = Field(description="Issues in cluster")
    total: int = Field(description="Total issues in cluster")
    limit: int = Field(description="Limit used")
    offset: int = Field(description="Offset used")
