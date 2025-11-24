"""API routes for Analyzer Service with Processing Manager."""

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from ...application.processing_manager import (
    ProcessingManager,
    get_processing_manager as get_global_processing_manager,
)
from ...application.use_cases.cluster_all_issues import ClusterAllIssuesUseCase
from ...application.use_cases.reprocess_issue import ReprocessIssueUseCase
from ...application.use_cases.search_similar_issues import SearchSimilarIssuesUseCase
from ...domain.repositories.cluster_repository import ClusterRepository
from ...domain.repositories.issue_repository import IssueRepository
from ...domain.repositories.preprocessed_issue_repository import (
    PreprocessedIssueRepository,
)
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
    get_text_preprocessor,
    get_vector_db_service,
)
from .schemas import (
    ClusterInfoResponse,
    ClusteringRequest,
    ClusteringResponse,
    ProcessingStatusResponse,
    ReprocessRequest,
    SearchRequest,
    SearchResponse,
    ServiceStatusResponse,
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
                label=cluster.label,
                size=cluster.size,
                description=cluster.description,
            )
            for cluster in clusters
        ]

    except Exception as e:
        logger.error(f"Failed to get clusters info: {e}", exc_info=True)
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
