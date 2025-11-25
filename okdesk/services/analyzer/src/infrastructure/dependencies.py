"""Dependency injection for FastAPI."""

from typing import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..domain.repositories.cluster_repository import ClusterRepository
from ..domain.repositories.issue_repository import IssueRepository
from ..domain.repositories.message_repository import MessageRepository
from ..domain.repositories.preprocessed_issue_repository import (
    PreprocessedIssueRepository,
)
from ..domain.repositories.stats_repository import StatsRepository
from ..domain.services.clustering_service import ClusteringService
from ..domain.services.embedding_generator import EmbeddingGenerator
from ..domain.services.text_preprocessor import TextPreprocessor
from ..domain.services.vector_db_service import VectorDBService
from .ml.sentence_transformer_wrapper import SentenceTransformerWrapper
from .ml.pymorphy_wrapper import PyMorphyWrapper
from .persistence.database import get_db_session
from .persistence.postgres import (
    ClusterRepositoryImpl,
    IssueRepositoryImpl,
    MessageRepositoryImpl,
    PreprocessedIssueRepositoryImpl,
)
from .persistence.postgres.stats_repository_impl import StatsRepositoryImpl
from .vectordb.chromadb_client import ChromaDBClient


# ==================== Database Session ====================
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session dependency.

    Yields:
        AsyncSession for database operations
    """
    async for session in get_db_session():
        yield session


# ==================== Repositories ====================
def get_issue_repository(
    session: AsyncSession = Depends(get_session),
) -> IssueRepository:
    """
    Get IssueRepository dependency.

    Args:
        session: Database session (injected by FastAPI)

    Returns:
        IssueRepository implementation
    """
    return IssueRepositoryImpl(session)


def get_preprocessed_issue_repository(
    session: AsyncSession = Depends(get_session),
) -> PreprocessedIssueRepository:
    """
    Get PreprocessedIssueRepository dependency.

    Args:
        session: Database session (injected by FastAPI)

    Returns:
        PreprocessedIssueRepository implementation
    """
    return PreprocessedIssueRepositoryImpl(session)


def get_message_repository(
    session: AsyncSession = Depends(get_session),
) -> MessageRepository:
    """
    Get MessageRepository dependency.

    Args:
        session: Database session (injected by FastAPI)

    Returns:
        MessageRepository implementation
    """
    return MessageRepositoryImpl(session)


def get_cluster_repository(
    session: AsyncSession = Depends(get_session),
) -> ClusterRepository:
    """
    Get ClusterRepository dependency.

    Args:
        session: Database session (injected by FastAPI)

    Returns:
        ClusterRepository implementation
    """
    return ClusterRepositoryImpl(session)


def get_stats_repository(
    session: AsyncSession = Depends(get_session),
) -> StatsRepository:
    """
    Get StatsRepository dependency.

    Args:
        session: Database session (injected by FastAPI)

    Returns:
        StatsRepository implementation
    """
    vectordb = get_vector_db_service()
    return StatsRepositoryImpl(session, vectordb)


# ==================== Domain Services ====================
def get_text_preprocessor() -> TextPreprocessor:
    """
    Get TextPreprocessor service with pymorphy2 initialized.

    Returns:
        TextPreprocessor instance with morph analyzer
    """
    preprocessor = TextPreprocessor()

    # TODO: PyMorphy2 is incompatible with Python 3.12 (uses deprecated inspect.getargspec)
    # Will work without lemmatization for now
    # Initialize pymorphy2 and inject into preprocessor
    # morph_wrapper = PyMorphyWrapper()
    # preprocessor.set_morph_analyzer(morph_wrapper.morph)

    return preprocessor


# Global singleton instance for EmbeddingGenerator
_embedding_generator: EmbeddingGenerator | None = None


def get_embedding_generator() -> EmbeddingGenerator:
    """
    Get EmbeddingGenerator service singleton.

    Returns:
        SentenceTransformer wrapper instance (singleton)
    """
    global _embedding_generator

    if _embedding_generator is None:
        _embedding_generator = SentenceTransformerWrapper(
            model_name=settings.embedding_model,
            device=settings.device,
        )

    return _embedding_generator


# Global singleton instance for VectorDBService
_vector_db_service: VectorDBService | None = None


def get_vector_db_service() -> VectorDBService:
    """
    Get VectorDBService singleton.

    Returns:
        ChromaDB client instance (singleton)
    """
    global _vector_db_service

    if _vector_db_service is None:
        _vector_db_service = ChromaDBClient(
            host=settings.chromadb_host,
            port=settings.chromadb_port,
            collection_name=settings.chromadb_collection,
        )

    return _vector_db_service


def get_clustering_service() -> ClusteringService:
    """
    Get ClusteringService.

    Returns:
        ClusteringService instance
    """
    return ClusteringService()


# ==================== Database Session (alias) ====================
async def get_database_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session (alias for get_session).

    Yields:
        AsyncSession for database operations
    """
    async for session in get_session():
        yield session


# ==================== Use Cases ====================
def get_process_batch_use_case(
    session: AsyncSession = Depends(get_session),
) -> "ProcessIssuesBatchUseCase":
    """
    Get ProcessIssuesBatchUseCase dependency.

    Args:
        session: Database session (injected by FastAPI)

    Returns:
        ProcessIssuesBatchUseCase instance
    """
    from ..application.use_cases.process_issues_batch import ProcessIssuesBatchUseCase

    # Create all dependencies
    issue_repo = IssueRepositoryImpl(session)
    preprocessed_repo = PreprocessedIssueRepositoryImpl(session)
    preprocessor = get_text_preprocessor()
    embedding_gen = get_embedding_generator()
    vectordb = get_vector_db_service()

    return ProcessIssuesBatchUseCase(
        issue_repo=issue_repo,
        preprocessed_repo=preprocessed_repo,
        preprocessor=preprocessor,
        embedding_gen=embedding_gen,
        vectordb=vectordb,
    )
