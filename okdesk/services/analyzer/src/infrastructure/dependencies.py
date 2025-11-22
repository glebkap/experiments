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


# ==================== Domain Services ====================
def get_text_preprocessor() -> TextPreprocessor:
    """
    Get TextPreprocessor service with pymorphy2 initialized.

    Returns:
        TextPreprocessor instance with morph analyzer
    """
    preprocessor = TextPreprocessor()

    # Initialize pymorphy2 and inject into preprocessor
    morph_wrapper = PyMorphyWrapper()
    preprocessor.set_morph_analyzer(morph_wrapper.morph)

    return preprocessor


def get_embedding_generator() -> EmbeddingGenerator:
    """
    Get EmbeddingGenerator service.

    Returns:
        SentenceTransformer wrapper instance
    """
    return SentenceTransformerWrapper(
        model_name=settings.embedding_model,
        device=settings.device,
    )


def get_vector_db_service() -> VectorDBService:
    """
    Get VectorDBService.

    Returns:
        ChromaDB client instance
    """
    return ChromaDBClient(
        host=settings.chromadb_host,
        port=settings.chromadb_port,
        collection_name=settings.chromadb_collection,
    )


def get_clustering_service() -> ClusteringService:
    """
    Get ClusteringService.

    Returns:
        ClusteringService instance
    """
    return ClusteringService()
