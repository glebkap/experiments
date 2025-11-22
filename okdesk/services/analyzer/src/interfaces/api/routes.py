"""API routes for Analyzer Service."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from ...application.use_cases.process_issues_batch import ProcessIssuesBatchUseCase
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
    PipelineRequest,
    PipelineResponse,
    ServiceStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ==================== Pipeline Endpoints ====================
@router.post("/pipeline/process", response_model=PipelineResponse)
async def process_batch(
    request: PipelineRequest,
    issue_repo: IssueRepository = Depends(get_issue_repository),
    preprocessed_repo: PreprocessedIssueRepository = Depends(
        get_preprocessed_issue_repository
    ),
    preprocessor: TextPreprocessor = Depends(get_text_preprocessor),
    embedding_gen: EmbeddingGenerator = Depends(get_embedding_generator),
    vectordb: VectorDBService = Depends(get_vector_db_service),
):
    """
    Process a batch of unprocessed issues through the pipeline.

    Stages:
    1. Fetch unprocessed issues from database
    2. Preprocess text (clean HTML, lemmatize)
    3. Generate embeddings
    4. Store in ChromaDB
    5. Mark as processed

    Args:
        request: Pipeline configuration (batch_size, device)

    Returns:
        Processing result with statistics
    """
    logger.info(
        f"Processing batch: batch_size={request.batch_size}, device={request.device}"
    )

    try:
        # Create use case
        use_case = ProcessIssuesBatchUseCase(
            issue_repo=issue_repo,
            preprocessed_repo=preprocessed_repo,
            preprocessor=preprocessor,
            embedding_gen=embedding_gen,
            vectordb=vectordb,
        )

        # Execute pipeline
        result = await use_case.execute(
            batch_size=request.batch_size, device=request.device
        )

        logger.info(
            f"Pipeline completed: processed={result.processed_count}, duration={result.duration_seconds:.2f}s"
        )

        return PipelineResponse(
            success=result.success,
            processed_count=result.processed_count,
            duration_seconds=result.duration_seconds,
            stats=result.stats,
            errors=result.errors,
        )

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipeline/status", response_model=ServiceStatusResponse)
async def get_pipeline_status(
    issue_repo: IssueRepository = Depends(get_issue_repository),
    cluster_repo: ClusterRepository = Depends(get_cluster_repository),
    vectordb: VectorDBService = Depends(get_vector_db_service),
):
    """
    Get current pipeline processing status.

    Returns:
        - Total issues in database
        - Unprocessed issues count
        - Total embeddings in ChromaDB
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
        logger.error(f"Failed to get pipeline status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Clustering Endpoints ====================
@router.post("/clustering/run")
async def run_clustering(
    method: str = "hdbscan",
    min_cluster_size: Optional[int] = None,
    n_clusters: Optional[int] = None,
    cluster_repo: ClusterRepository = Depends(get_cluster_repository),
    vectordb: VectorDBService = Depends(get_vector_db_service),
    clustering_service: ClusteringService = Depends(get_clustering_service),
):
    """
    Run clustering on all embeddings in ChromaDB.

    Args:
        method: Clustering method ('hdbscan' or 'kmeans')
        min_cluster_size: Minimum cluster size for HDBSCAN
        n_clusters: Number of clusters for K-means (auto-detect if None)

    Returns:
        Clustering result with cluster count and assignments
    """
    logger.info(f"Running {method} clustering")

    try:
        # Get all embeddings from ChromaDB
        issue_ids, embeddings = await vectordb.get_all_embeddings()

        if len(embeddings) == 0:
            raise HTTPException(
                status_code=400, detail="No embeddings found in ChromaDB"
            )

        # Run clustering
        if method == "hdbscan":
            min_size = min_cluster_size or 5
            labels, centroids = clustering_service.cluster_hdbscan(
                embeddings, min_cluster_size=min_size
            )
        elif method == "kmeans":
            labels, centroids = clustering_service.cluster_kmeans(
                embeddings, n_clusters=n_clusters
            )
        else:
            raise HTTPException(
                status_code=400, detail=f"Unknown clustering method: {method}"
            )

        # Compute distances
        distances = clustering_service.compute_distances(embeddings, centroids, labels)

        # Clear existing clusters
        await cluster_repo.clear_all_clusters()

        # Create clusters and assign messages
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Outliers

        cluster_map = {}
        for label in sorted(unique_labels):
            cluster = await cluster_repo.create_cluster(
                label=f"Cluster {label}",
                centroid_embedding=centroids[label],
            )
            cluster_map[label] = cluster.id

        # Assign messages to clusters
        for label in cluster_map:
            mask = labels == label
            msg_ids = [issue_ids[i] for i, m in enumerate(mask) if m]
            msg_distances = [distances[i] for i, m in enumerate(mask) if m]

            await cluster_repo.assign_messages(
                cluster_id=cluster_map[label],
                message_ids=msg_ids,
                distances=msg_distances,
            )

        logger.info(f"Clustering completed: {len(cluster_map)} clusters created")

        return {
            "success": True,
            "method": method,
            "total_items": len(embeddings),
            "num_clusters": len(cluster_map),
            "outliers": int((labels == -1).sum()),
        }

    except HTTPException:
        raise
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
        List of cluster information (ID, label, size)
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
@router.post("/search/similar")
async def search_similar_issues(
    query: str,
    top_k: int = 10,
    min_similarity: float = 0.7,
    preprocessor: TextPreprocessor = Depends(get_text_preprocessor),
    embedding_gen: EmbeddingGenerator = Depends(get_embedding_generator),
    vectordb: VectorDBService = Depends(get_vector_db_service),
):
    """
    Search for similar issues using semantic similarity.

    Args:
        query: Search query text
        top_k: Number of results to return
        min_similarity: Minimum similarity threshold (0-1)

    Returns:
        List of similar issues with similarity scores
    """
    logger.info(f"Searching for similar issues: query='{query[:50]}...', top_k={top_k}")

    try:
        # Preprocess query
        preprocessed_query = preprocessor.preprocess(query)

        if not preprocessed_query:
            raise HTTPException(
                status_code=400, detail="Query is empty after preprocessing"
            )

        # Generate embedding
        query_embedding = embedding_gen.generate_single(preprocessed_query)

        # Search in ChromaDB
        results = await vectordb.search_similar(
            query_embedding=query_embedding,
            top_k=top_k,
            min_similarity=min_similarity,
        )

        logger.info(f"Found {len(results)} similar issues")

        return {
            "query": query,
            "results": [
                {
                    "issue_id": str(issue_id),
                    "similarity": float(similarity),
                    "text_snippet": text[:200] + "..." if len(text) > 200 else text,
                }
                for issue_id, similarity, text in results
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
