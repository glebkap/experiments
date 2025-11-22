"""Use case for searching similar issues using semantic search."""

import logging
from typing import List
from uuid import UUID

from ...domain.repositories.issue_repository import IssueRepository
from ...domain.services.embedding_generator import EmbeddingGenerator
from ...domain.services.text_preprocessor import TextPreprocessor
from ...domain.services.vector_db_service import VectorDBService
from ..dto.similar_issue_dto import SimilarIssueDTO

logger = logging.getLogger(__name__)


class SearchSimilarIssuesUseCase:
    """Use case for finding similar issues using semantic similarity."""

    def __init__(
        self,
        vectordb: VectorDBService,
        issue_repo: IssueRepository,
        preprocessor: TextPreprocessor,
        embedding_gen: EmbeddingGenerator,
    ):
        """
        Initialize use case with dependencies.

        Args:
            vectordb: Vector database service
            issue_repo: Issue repository
            preprocessor: Text preprocessor
            embedding_gen: Embedding generator
        """
        self.vectordb = vectordb
        self.issue_repo = issue_repo
        self.preprocessor = preprocessor
        self.embedding_gen = embedding_gen

    async def execute_by_id(
        self, issue_id: UUID, top_k: int = 10, min_similarity: float = 0.7
    ) -> List[SimilarIssueDTO]:
        """
        Find similar issues by issue ID.

        Args:
            issue_id: Issue UUID to find similar issues for
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of similar issues with similarity scores

        Raises:
            ValueError: If issue not found
        """
        logger.info(f"Searching similar issues for {issue_id}, top_k={top_k}")

        # Get the issue
        issue = await self.issue_repo.get_by_id(issue_id)
        if not issue:
            raise ValueError(f"Issue {issue_id} not found")

        # Preprocess and generate embedding
        text = issue.get_combined_text()
        preprocessed = self.preprocessor.preprocess(text)

        if not preprocessed:
            logger.warning(f"Empty text after preprocessing for issue {issue_id}")
            return []

        # Generate query embedding
        query_embedding = self.embedding_gen.generate_single(preprocessed)

        # Search in ChromaDB
        results = await self.vectordb.search_similar(
            query_embedding=query_embedding,
            top_k=top_k + 1,  # +1 to exclude the issue itself
            min_similarity=min_similarity,
        )

        # Filter out the issue itself and convert to DTOs
        similar_issues = []
        for similar_id, similarity, text_snippet in results:
            if similar_id == issue_id:
                continue  # Skip the issue itself

            similar_issue = await self.issue_repo.get_by_id(similar_id)
            if similar_issue:
                similar_issues.append(
                    SimilarIssueDTO(
                        issue_id=str(similar_issue.id),
                        title=similar_issue.title or "",
                        description=similar_issue.description or "",
                        similarity_score=float(similarity),
                    )
                )

            if len(similar_issues) >= top_k:
                break

        logger.info(f"Found {len(similar_issues)} similar issues for {issue_id}")

        return similar_issues

    async def execute_by_text(
        self, query_text: str, top_k: int = 10, min_similarity: float = 0.7
    ) -> List[SimilarIssueDTO]:
        """
        Find similar issues by free-form text query.

        Args:
            query_text: Search query text
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of similar issues with similarity scores

        Raises:
            ValueError: If query is empty after preprocessing
        """
        logger.info(f"Searching similar issues for query: '{query_text[:50]}...'")

        # Preprocess query
        preprocessed = self.preprocessor.preprocess(query_text)

        if not preprocessed:
            raise ValueError("Query is empty after preprocessing")

        # Generate query embedding
        query_embedding = self.embedding_gen.generate_single(preprocessed)

        # Search in ChromaDB
        results = await self.vectordb.search_similar(
            query_embedding=query_embedding,
            top_k=top_k,
            min_similarity=min_similarity,
        )

        # Convert to DTOs
        similar_issues = []
        for issue_id, similarity, text_snippet in results:
            issue = await self.issue_repo.get_by_id(issue_id)
            if issue:
                similar_issues.append(
                    SimilarIssueDTO(
                        issue_id=str(issue.id),
                        title=issue.title or "",
                        description=issue.description or "",
                        similarity_score=float(similarity),
                    )
                )

        logger.info(f"Found {len(similar_issues)} similar issues for query")

        return similar_issues
