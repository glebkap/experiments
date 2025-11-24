"""Main FastAPI application."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """Initialize services on startup."""
    logger.info("=" * 60)
    logger.info(f"Starting {settings.api_title} v{settings.api_version}")
    logger.info(f"Database: {settings.database_url.split('@')[1] if '@' in settings.database_url else settings.database_url}")
    logger.info(f"ChromaDB: {settings.chromadb_host}:{settings.chromadb_port}")
    logger.info(f"Embedding model: {settings.embedding_model}")
    logger.info(f"Device: {settings.device}")
    logger.info("=" * 60)

    # Initialize ProcessingManager
    from .application.processing_manager import set_processing_manager, ProcessingManager
    from .infrastructure.dependencies import (
        get_process_batch_use_case,
        get_preprocessed_issue_repository,
        get_vector_db_service,
        get_database_session,
    )

    # Create dependencies
    async for session in get_database_session():
        process_batch_use_case = get_process_batch_use_case(session)

        # Handle reprocess_all flag - clear all preprocessed issues and ChromaDB
        if settings.reprocess_all:
            logger.warning("=" * 70)
            logger.warning("⚠️  REPROCESS_ALL flag is enabled!")
            logger.warning("⚠️  This will clear all preprocessed issues AND ChromaDB embeddings!")
            logger.warning("⚠️  Force reprocessing with current embedding model settings!")
            logger.warning("=" * 70)

            # Clear preprocessed_issues table
            preprocessed_repo = get_preprocessed_issue_repository(session)
            deleted_issues = await preprocessed_repo.clear_all()
            logger.warning(f"✅ Cleared {deleted_issues} preprocessed issues from PostgreSQL")

            # Clear ChromaDB collection
            vectordb = get_vector_db_service()
            deleted_embeddings = vectordb.clear_all()
            logger.warning(f"✅ Cleared {deleted_embeddings} embeddings from ChromaDB")

            logger.warning("=" * 70)

        # Create and register ProcessingManager
        manager = ProcessingManager(
            process_batch_use_case=process_batch_use_case,
            batch_size=settings.batch_size,
            poll_interval_seconds=settings.poll_interval_seconds,
            device=settings.device,
        )
        set_processing_manager(manager)

        logger.info(
            f"ProcessingManager initialized (batch_size={settings.batch_size}, "
            f"poll_interval={settings.poll_interval_seconds}s, device={settings.device})"
        )

        # Auto-start if configured
        if settings.auto_start_processing:
            await manager.start()
            logger.info("Background processing auto-started")
        else:
            logger.info("Background processing NOT auto-started (use POST /processing/start)")

        break  # Only need one iteration


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("Shutting down Analyzer Service")

    # Stop ProcessingManager gracefully
    from .application.processing_manager import get_processing_manager

    manager = get_processing_manager()
    if manager:
        await manager.stop()
        logger.info("ProcessingManager stopped")
    else:
        logger.warning("ProcessingManager not initialized, skipping shutdown")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.api_title,
        "version": settings.api_version,
    }


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": settings.api_title,
        "version": settings.api_version,
        "description": settings.api_description,
        "docs": "/docs",
        "health": "/health",
    }


# API routes
from .interfaces.api.routes_new import router

app.include_router(router, prefix="/api/v1/analyzer", tags=["analyzer"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
