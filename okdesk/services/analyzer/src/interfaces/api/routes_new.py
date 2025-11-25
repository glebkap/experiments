"""API routes for Analyzer Service with Processing Manager."""

import csv
import io
import json
import logging
from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from ...application.processing_manager import (
    ProcessingManager,
    get_processing_manager as get_global_processing_manager,
)
from ...application.use_cases.cluster_all_issues import ClusterAllIssuesUseCase
from ...application.use_cases.get_cluster_issues import GetClusterIssuesUseCase
from ...application.use_cases.reprocess_issue import ReprocessIssueUseCase
from ...application.use_cases.search_similar_issues import SearchSimilarIssuesUseCase
from ...domain.repositories.cluster_repository import ClusterRepository
from ...domain.repositories.issue_repository import IssueRepository
from ...domain.repositories.preprocessed_issue_repository import (
    PreprocessedIssueRepository,
)
from ...domain.repositories.stats_repository import StatsRepository
from ...domain.services.clustering_service import ClusteringService
from ...domain.services.embedding_generator import EmbeddingGenerator
from ...domain.services.text_preprocessor import TextPreprocessor
from ...domain.services.vector_db_service import VectorDBService
from ...infrastructure.dependencies import (
    get_cluster_repository,
    get_clustering_service,
    get_embedding_generator,
    get_issue_repository,
    get_preprocessed_issue_repository,
    get_stats_repository,
    get_text_preprocessor,
    get_vector_db_service,
)
from .schemas import (
    ClusterInfoResponse,
    ClusteringRequest,
    ClusteringResponse,
    ClusterIssuesResponseSchema,
    ClusterStatsSchema,
    ClustersStatsResponseSchema,
    ExportRequestSchema,
    FulltextSearchResponseSchema,
    IssueDetailSchema,
    IssueListItemSchema,
    IssueListResponseSchema,
    MessageSchema,
    ProcessingStatsSchema,
    ProcessingStatusResponse,
    ReprocessRequest,
    SearchRequest,
    SearchResponse,
    ServiceStatusResponse,
    SourceStatsSchema,
    SourcesStatsResponseSchema,
    TimelinePointSchema,
    TimelineStatsResponseSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def get_processing_manager() -> ProcessingManager:
    """
    Get global processing manager instance.

    Returns:
        ProcessingManager singleton
    """
    manager = get_global_processing_manager()
    if manager is None:
        raise HTTPException(
            status_code=500, detail="Processing manager not initialized"
        )
    return manager


# ==================== Processing Control Endpoints ====================
@router.post("/processing/start", response_model=ProcessingStatusResponse)
async def start_processing(
    manager: ProcessingManager = Depends(get_processing_manager),
):
    """
    Start background processing of issues.

    The processor will continuously poll for new unprocessed issues
    every `poll_interval_seconds` and process them in batches.

    Returns:
        Processing status
    """
    try:
        status = await manager.start()
        logger.info("Background processing started")
        return ProcessingStatusResponse(**status)

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/processing/pause", response_model=ProcessingStatusResponse)
async def pause_processing(
    manager: ProcessingManager = Depends(get_processing_manager),
):
    """
    Pause background processing.

    Processing can be resumed with /processing/resume endpoint.

    Returns:
        Processing status
    """
    try:
        status = await manager.pause()
        logger.info("Background processing paused")
        return ProcessingStatusResponse(**status)

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to pause processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/processing/resume", response_model=ProcessingStatusResponse)
async def resume_processing(
    manager: ProcessingManager = Depends(get_processing_manager),
):
    """
    Resume background processing after pause.

    Returns:
        Processing status
    """
    try:
        status = await manager.resume()
        logger.info("Background processing resumed")
        return ProcessingStatusResponse(**status)

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to resume processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/processing/stop", response_model=ProcessingStatusResponse)
async def stop_processing(
    manager: ProcessingManager = Depends(get_processing_manager),
):
    """
    Stop background processing.

    Returns:
        Final processing status
    """
    try:
        status = await manager.stop()
        logger.info("Background processing stopped")
        return ProcessingStatusResponse(**status)

    except Exception as e:
        logger.error(f"Failed to stop processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/processing/status", response_model=ProcessingStatusResponse)
async def get_processing_status(
    manager: ProcessingManager = Depends(get_processing_manager),
):
    """
    Get current processing status.

    Returns:
        - Current state (stopped/running/paused)
        - Total issues processed
        - Uptime
        - Last batch timestamp
    """
    status = manager.get_status()
    return ProcessingStatusResponse(**status)


@router.post("/processing/process-batch")
async def process_single_batch(
    manager: ProcessingManager = Depends(get_processing_manager),
):
    """
    Process a single batch of issues manually.

    This endpoint processes one batch regardless of the manager state.
    Useful for manual processing or testing.

    Returns:
        Processing result with statistics
    """
    try:
        result = await manager.process_single_batch()
        logger.info(f"Manual batch processed: {result['processed_count']} issues")
        return result

    except Exception as e:
        logger.error(f"Failed to process batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Issue Management Endpoints ====================
@router.post("/issues/{issue_id}/reprocess")
async def reprocess_issue(
    issue_id: UUID,
    issue_repo: IssueRepository = Depends(get_issue_repository),
    preprocessed_repo: PreprocessedIssueRepository = Depends(
        get_preprocessed_issue_repository
    ),
    vectordb: VectorDBService = Depends(get_vector_db_service),
    preprocessor: TextPreprocessor = Depends(get_text_preprocessor),
    embedding_gen: EmbeddingGenerator = Depends(get_embedding_generator),
):
    """
    Reprocess a single issue.

    Deletes existing preprocessed data and embeddings,
    then re-runs the issue through the processing pipeline.

    Args:
        issue_id: Issue UUID to reprocess

    Returns:
        Reprocessing result
    """
    logger.info(f"Reprocessing issue {issue_id}")

    try:
        use_case = ReprocessIssueUseCase(
            issue_repo=issue_repo,
            preprocessed_repo=preprocessed_repo,
            vectordb=vectordb,
            process_batch_uc=get_processing_manager().process_batch_use_case,
        )

        result = await use_case.execute(issue_id)

        return {
            "success": result.success,
            "issue_id": str(issue_id),
            "processed_count": result.processed_count,
            "duration_seconds": result.duration_seconds,
            "stats": result.stats,
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to reprocess issue {issue_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Clustering Endpoints ====================
@router.post("/clustering/run", response_model=ClusteringResponse)
async def run_clustering(
    request: ClusteringRequest,
    vectordb: VectorDBService = Depends(get_vector_db_service),
    clustering_service: ClusteringService = Depends(get_clustering_service),
    cluster_repo: ClusterRepository = Depends(get_cluster_repository),
    issue_repo: IssueRepository = Depends(get_issue_repository),
):
    """
    Run clustering on all processed issues.

    Args:
        request: Clustering configuration

    Returns:
        Clustering result with cluster information
    """
    logger.info(f"Running {request.method} clustering")

    try:
        use_case = ClusterAllIssuesUseCase(
            vectordb=vectordb,
            clustering_service=clustering_service,
            cluster_repo=cluster_repo,
            issue_repo=issue_repo,
        )

        result = await use_case.execute(
            method=request.method,
            min_cluster_size=request.min_cluster_size,
            min_samples=request.min_samples,
            n_clusters=request.n_clusters,
        )

        return ClusteringResponse(
            success=True,
            total_issues=result.total_issues,
            num_clusters=result.num_clusters,
            outliers_count=result.outliers_count,
            duration_seconds=result.duration_seconds,
            clusters=result.clusters,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Clustering failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/clustering/info", response_model=list[ClusterInfoResponse])
async def get_clusters_info(
    cluster_repo: ClusterRepository = Depends(get_cluster_repository),
):
    """
    Get information about all clusters.

    Returns:
        List of cluster information
    """
    try:
        clusters = await cluster_repo.get_all_clusters()

        return [
            ClusterInfoResponse(
                id=str(cluster.id),
                cluster_label=cluster.cluster_label,
                name=cluster.name,
                size=cluster.size,
                description=cluster.description,
            )
            for cluster in clusters
        ]

    except Exception as e:
        logger.error(f"Failed to get clusters info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/clustering/{cluster_id}/issues")
async def get_cluster_issues(
    cluster_id: UUID,
    limit: Optional[int] = None,
    offset: int = 0,
    cluster_repo: ClusterRepository = Depends(get_cluster_repository),
    issue_repo: IssueRepository = Depends(get_issue_repository),
):
    """
    Get all issues in a specific cluster.

    Args:
        cluster_id: Cluster UUID
        limit: Maximum number of issues to return (None = all)
        offset: Number of issues to skip for pagination

    Returns:
        List of issues with full information
    """
    try:
        use_case = GetClusterIssuesUseCase(
            cluster_repo=cluster_repo,
            issue_repo=issue_repo,
        )

        issues = await use_case.execute(
            cluster_id=cluster_id,
            limit=limit,
            offset=offset,
        )

        return {
            "cluster_id": str(cluster_id),
            "total": len(issues),
            "limit": limit,
            "offset": offset,
            "issues": [issue.to_dict() for issue in issues],
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get cluster issues: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Search Endpoints ====================
@router.post("/search/similar", response_model=SearchResponse)
async def search_similar_issues(
    request: SearchRequest,
    vectordb: VectorDBService = Depends(get_vector_db_service),
    issue_repo: IssueRepository = Depends(get_issue_repository),
    preprocessor: TextPreprocessor = Depends(get_text_preprocessor),
    embedding_gen: EmbeddingGenerator = Depends(get_embedding_generator),
):
    """
    Search for similar issues using semantic similarity.

    Can search by:
    - issue_id: Find issues similar to a specific issue
    - query: Find issues similar to free-form text

    Args:
        request: Search request (either issue_id or query required)

    Returns:
        List of similar issues with similarity scores
    """
    logger.info(f"Searching similar issues: {request}")

    try:
        use_case = SearchSimilarIssuesUseCase(
            vectordb=vectordb,
            issue_repo=issue_repo,
            preprocessor=preprocessor,
            embedding_gen=embedding_gen,
        )

        if request.issue_id:
            results = await use_case.execute_by_id(
                issue_id=request.issue_id,
                top_k=request.top_k,
                min_similarity=request.min_similarity,
            )
        elif request.query:
            results = await use_case.execute_by_text(
                query_text=request.query,
                top_k=request.top_k,
                min_similarity=request.min_similarity,
            )
        else:
            raise HTTPException(
                status_code=400, detail="Either issue_id or query must be provided"
            )

        return SearchResponse(
            query=request.query or f"issue:{request.issue_id}",
            results=results,
            total_results=len(results),
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Service Status Endpoint ====================
@router.get("/status", response_model=ServiceStatusResponse)
async def get_service_status(
    issue_repo: IssueRepository = Depends(get_issue_repository),
    cluster_repo: ClusterRepository = Depends(get_cluster_repository),
    vectordb: VectorDBService = Depends(get_vector_db_service),
):
    """
    Get overall service status.

    Returns:
        - Total issues in database
        - Unprocessed issues count
        - Total embeddings
        - Total clusters
    """
    try:
        total_issues = await issue_repo.count_total()
        unprocessed_count = await issue_repo.count_unprocessed()
        total_clusters = await cluster_repo.count_clusters()

        # Get ChromaDB info
        collection_info = vectordb.get_collection_info()
        embeddings_count = collection_info.get("count", 0)

        return ServiceStatusResponse(
            total_issues=total_issues,
            unprocessed_issues=unprocessed_count,
            processed_issues=total_issues - unprocessed_count,
            total_embeddings=embeddings_count,
            total_clusters=total_clusters,
        )

    except Exception as e:
        logger.error(f"Failed to get service status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Issues Endpoints (04-analyzer-extensions) ====================
@router.get("/issues", response_model=IssueListResponseSchema)
async def get_issues(
    status: Optional[str] = Query(None, description="Filter by status"),
    source_id: Optional[UUID] = Query(None, description="Filter by source"),
    priority: Optional[int] = Query(None, ge=1, le=4, description="Filter by priority"),
    date_from: Optional[str] = Query(None, description="Filter from date (ISO)"),
    date_to: Optional[str] = Query(None, description="Filter to date (ISO)"),
    limit: int = Query(50, ge=1, le=100, description="Max results"),
    offset: int = Query(0, ge=0, description="Results to skip"),
    issue_repo: IssueRepository = Depends(get_issue_repository),
):
    """
    Get list of issues with filtering and pagination.

    Query Parameters:
        - status: Filter by issue status (opened/wait/completed/closed)
        - source_id: Filter by source UUID
        - priority: Filter by priority (1-4)
        - date_from, date_to: Filter by creation date range (ISO format)
        - limit, offset: Pagination
    """
    try:
        # Parse dates if provided
        parsed_date_from = None
        parsed_date_to = None
        if date_from:
            parsed_date_from = datetime.fromisoformat(date_from.replace("Z", "+00:00"))
        if date_to:
            parsed_date_to = datetime.fromisoformat(date_to.replace("Z", "+00:00"))

        issues = await issue_repo.get_issues_with_filters(
            status=status,
            source_id=source_id,
            priority=priority,
            date_from=parsed_date_from,
            date_to=parsed_date_to,
            limit=limit,
            offset=offset,
        )

        total = await issue_repo.count_issues_with_filters(
            status=status,
            source_id=source_id,
            priority=priority,
            date_from=parsed_date_from,
            date_to=parsed_date_to,
        )

        items = [
            IssueListItemSchema(
                id=str(issue.id),
                external_id=issue.external_id,
                title=issue.title,
                status=issue.status,
                priority=issue.priority,
                created_at=issue.created_at.isoformat(),
            )
            for issue in issues
        ]

        return IssueListResponseSchema(
            items=items,
            total=total,
            limit=limit,
            offset=offset,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
    except Exception as e:
        logger.error(f"Failed to get issues: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/issues/{issue_id}", response_model=IssueDetailSchema)
async def get_issue_detail(
    issue_id: UUID,
    issue_repo: IssueRepository = Depends(get_issue_repository),
    cluster_repo: ClusterRepository = Depends(get_cluster_repository),
):
    """
    Get detailed information about a specific issue with messages.
    """
    try:
        # Get issue with messages
        result = await issue_repo.get_issue_with_messages(issue_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Issue {issue_id} not found")

        issue, messages = result

        # Get source info
        source_info = await issue_repo.get_issue_source_info(issue_id)

        # Get cluster info
        cluster_info = await cluster_repo.get_issue_cluster(issue_id)

        messages_schema = [
            MessageSchema(
                id=str(m.id),
                external_id=m.external_id,
                author_name=m.author_name,
                author_type=m.author_type,
                content=m.content,
                is_public=m.is_public,
                published_at=m.published_at.isoformat() if m.published_at else None,
            )
            for m in messages
        ]

        return IssueDetailSchema(
            id=str(issue.id),
            external_id=issue.external_id,
            title=issue.title,
            description=issue.description,
            status=issue.status,
            priority=issue.priority,
            created_at=issue.created_at.isoformat(),
            updated_at=issue.updated_at.isoformat() if issue.updated_at else None,
            completed_at=None,  # Not in our Issue model
            source_name=source_info["source_name"] if source_info else None,
            source_type=source_info["source_type"] if source_info else None,
            messages=messages_schema,
            cluster_id=str(cluster_info[0]) if cluster_info else None,
            cluster_label=cluster_info[1] if cluster_info else None,
            distance_to_centroid=cluster_info[2] if cluster_info else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get issue detail: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Enhanced Cluster Issues Endpoint ====================
@router.get("/clusters/{cluster_id}/issues", response_model=ClusterIssuesResponseSchema)
async def get_cluster_issues_extended(
    cluster_id: UUID,
    limit: int = Query(50, ge=1, le=100, description="Max results"),
    offset: int = Query(0, ge=0, description="Results to skip"),
    cluster_repo: ClusterRepository = Depends(get_cluster_repository),
):
    """
    Get issues in a cluster with full issue information.

    Issues are sorted by distance to centroid (closest first).
    """
    try:
        result = await cluster_repo.get_cluster_with_issues(
            cluster_id=cluster_id,
            limit=limit,
            offset=offset,
        )

        if result is None:
            raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")

        cluster, issues, total = result

        items = [
            IssueListItemSchema(
                id=str(issue.id),
                external_id=issue.external_id,
                title=issue.title,
                status=issue.status,
                priority=issue.priority,
                created_at=issue.created_at.isoformat(),
            )
            for issue in issues
        ]

        return ClusterIssuesResponseSchema(
            cluster_id=str(cluster.id),
            cluster_label=cluster.cluster_label,
            cluster_name=cluster.name,
            items=items,
            total=total,
            limit=limit,
            offset=offset,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get cluster issues: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Full-text Search Endpoint ====================
@router.get("/search/fulltext", response_model=FulltextSearchResponseSchema)
async def fulltext_search(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(50, ge=1, le=100, description="Max results"),
    offset: int = Query(0, ge=0, description="Results to skip"),
    issue_repo: IssueRepository = Depends(get_issue_repository),
):
    """
    Full-text search in preprocessed issues using PostgreSQL FTS.

    Searches in the preprocessed content of issues.
    Results are sorted by relevance.
    """
    try:
        issues, total = await issue_repo.fulltext_search(
            query=q,
            limit=limit,
            offset=offset,
        )

        items = [
            IssueListItemSchema(
                id=str(issue.id),
                external_id=issue.external_id,
                title=issue.title,
                status=issue.status,
                priority=issue.priority,
                created_at=issue.created_at.isoformat(),
            )
            for issue in issues
        ]

        return FulltextSearchResponseSchema(
            query=q,
            items=items,
            total=total,
            limit=limit,
            offset=offset,
        )

    except Exception as e:
        logger.error(f"Full-text search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Stats Endpoints (04-analyzer-extensions) ====================
@router.get("/stats/processing", response_model=ProcessingStatsSchema)
async def get_processing_stats(
    stats_repo: StatsRepository = Depends(get_stats_repository),
):
    """
    Get overall processing statistics.
    """
    try:
        stats = await stats_repo.get_processing_stats()
        return ProcessingStatsSchema(**stats)

    except Exception as e:
        logger.error(f"Failed to get processing stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/sources", response_model=SourcesStatsResponseSchema)
async def get_sources_stats(
    stats_repo: StatsRepository = Depends(get_stats_repository),
):
    """
    Get statistics grouped by data source.
    """
    try:
        stats = await stats_repo.get_sources_stats()
        sources = [SourceStatsSchema(**s) for s in stats]
        return SourcesStatsResponseSchema(sources=sources)

    except Exception as e:
        logger.error(f"Failed to get sources stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/clusters", response_model=ClustersStatsResponseSchema)
async def get_clusters_stats(
    cluster_repo: ClusterRepository = Depends(get_cluster_repository),
):
    """
    Get statistics for all clusters.
    """
    try:
        stats = await cluster_repo.get_cluster_stats()
        clusters = [ClusterStatsSchema(**s) for s in stats]
        return ClustersStatsResponseSchema(
            clusters=clusters,
            total_clusters=len(clusters),
        )

    except Exception as e:
        logger.error(f"Failed to get clusters stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/timeline", response_model=TimelineStatsResponseSchema)
async def get_timeline_stats(
    date_from: str = Query(..., description="Start date (ISO format)"),
    date_to: str = Query(..., description="End date (ISO format)"),
    group_by: str = Query("day", description="Grouping: day, week, month"),
    stats_repo: StatsRepository = Depends(get_stats_repository),
):
    """
    Get timeline statistics for issue creation and processing.
    """
    try:
        # Parse dates
        parsed_date_from = datetime.fromisoformat(date_from.replace("Z", "+00:00"))
        parsed_date_to = datetime.fromisoformat(date_to.replace("Z", "+00:00"))

        if group_by not in ("day", "week", "month"):
            raise HTTPException(
                status_code=400,
                detail="group_by must be one of: day, week, month",
            )

        stats = await stats_repo.get_timeline_stats(
            date_from=parsed_date_from,
            date_to=parsed_date_to,
            group_by=group_by,
        )

        points = [TimelinePointSchema(**s) for s in stats]

        return TimelineStatsResponseSchema(
            date_from=date_from,
            date_to=date_to,
            group_by=group_by,
            points=points,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get timeline stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Export Endpoint (04-analyzer-extensions) ====================
@router.post("/export")
async def export_data(
    request: ExportRequestSchema,
    issue_repo: IssueRepository = Depends(get_issue_repository),
):
    """
    Export issues to CSV or JSON format.

    Body:
        - format: "csv" or "json"
        - filters: Optional filters (same as GET /issues)

    Returns:
        StreamingResponse with file download
    """
    try:
        # Parse filters
        filters = request.filters
        parsed_date_from = None
        parsed_date_to = None

        if filters:
            if filters.date_from:
                parsed_date_from = datetime.fromisoformat(
                    filters.date_from.replace("Z", "+00:00")
                )
            if filters.date_to:
                parsed_date_to = datetime.fromisoformat(
                    filters.date_to.replace("Z", "+00:00")
                )

        # Get all issues matching filters (with high limit)
        issues = await issue_repo.get_issues_with_filters(
            status=filters.status if filters else None,
            source_id=filters.source_id if filters else None,
            priority=filters.priority if filters else None,
            date_from=parsed_date_from,
            date_to=parsed_date_to,
            limit=10000,  # High limit for export
            offset=0,
        )

        if request.format == "csv":
            # Generate CSV
            output = io.StringIO()
            writer = csv.writer(output)

            # Header
            writer.writerow([
                "id", "external_id", "title", "status", "priority", "created_at"
            ])

            # Data
            for issue in issues:
                writer.writerow([
                    str(issue.id),
                    issue.external_id,
                    issue.title or "",
                    issue.status,
                    issue.priority or "",
                    issue.created_at.isoformat(),
                ])

            output.seek(0)
            content = output.getvalue()

            return StreamingResponse(
                iter([content]),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=issues_export.csv"
                },
            )

        else:  # JSON
            data = [
                {
                    "id": str(issue.id),
                    "external_id": issue.external_id,
                    "title": issue.title,
                    "status": issue.status,
                    "priority": issue.priority,
                    "created_at": issue.created_at.isoformat(),
                }
                for issue in issues
            ]

            content = json.dumps(data, ensure_ascii=False, indent=2)

            return StreamingResponse(
                iter([content]),
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename=issues_export.json"
                },
            )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {e}")
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
