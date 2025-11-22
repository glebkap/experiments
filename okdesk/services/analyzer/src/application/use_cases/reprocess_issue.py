"""Use case for reprocessing a single issue."""

import logging
from uuid import UUID

from ...domain.repositories.issue_repository import IssueRepository
from ...domain.repositories.preprocessed_issue_repository import PreprocessedIssueRepository
from ...domain.services.vector_db_service import VectorDBService
from ..dto.processing_result_dto import ProcessingResultDTO
from .process_issues_batch import ProcessIssuesBatchUseCase

logger = logging.getLogger(__name__)


class ReprocessIssueUseCase:
    """Use case for reprocessing a single issue through the pipeline."""

    def __init__(
        self,
        issue_repo: IssueRepository,
        preprocessed_repo: PreprocessedIssueRepository,
        vectordb: VectorDBService,
        process_batch_uc: ProcessIssuesBatchUseCase,
    ):
        """
        Initialize use case with dependencies.

        Args:
            issue_repo: Issue repository
            preprocessed_repo: Preprocessed issue repository
            vectordb: Vector database service
            process_batch_uc: Batch processing use case
        """
        self.issue_repo = issue_repo
        self.preprocessed_repo = preprocessed_repo
        self.vectordb = vectordb
        self.process_batch_uc = process_batch_uc

    async def execute(self, issue_id: UUID) -> ProcessingResultDTO:
        """
        Reprocess a specific issue.

        This will delete existing preprocessed data and embeddings,
        then run the issue through the pipeline again.

        Args:
            issue_id: UUID of the issue to reprocess

        Returns:
            Processing result with statistics

        Raises:
            ValueError: If issue not found
        """
        logger.info(f"Reprocessing issue {issue_id}")

        # Verify issue exists
        issue = await self.issue_repo.get_by_id(issue_id)
        if not issue:
            raise ValueError(f"Issue {issue_id} not found")

        # Delete from preprocessed_issues table
        await self.preprocessed_repo.delete(issue_id)
        logger.info(f"Deleted preprocessed data for issue {issue_id}")

        # Delete from ChromaDB
        await self.vectordb.delete_by_issue_id(issue_id)
        logger.info(f"Deleted embedding for issue {issue_id}")

        # Reprocess through pipeline (batch_size=1 to process just this issue)
        result = await self.process_batch_uc.execute(batch_size=1, device="cpu")

        logger.info(f"Reprocessed issue {issue_id}: {result.processed_count} issues")

        return result
