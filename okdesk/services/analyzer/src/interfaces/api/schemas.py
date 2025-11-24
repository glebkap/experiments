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
    label: str = Field(description="Cluster label")
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
