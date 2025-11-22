"""Use case for processing a batch of issues through pipeline."""

import logging

from ...domain.repositories.issue_repository import IssueRepository
from ...domain.repositories.preprocessed_issue_repository import PreprocessedIssueRepository
from ...domain.services.embedding_generator import EmbeddingGenerator
from ...domain.services.text_preprocessor import TextPreprocessor
from ...domain.services.vector_db_service import VectorDBService
from ..dto.processing_result_dto import ProcessingResultDTO
from ..pipeline.pipeline_config import PipelineConfig
from ..pipeline.pipeline_executor import PipelineExecutor
from ..pipeline.stages.stage_0_fetch import Stage0FetchIssues
from ..pipeline.stages.stage_1_preprocess import Stage1Preprocess
from ..pipeline.stages.stage_2_embed import Stage2GenerateEmbeddings
from ..pipeline.stages.stage_3_vectordb import Stage3VectorDBStorage
from ..pipeline.stages.stage_4_complete import Stage4Complete

logger = logging.getLogger(__name__)


class ProcessIssuesBatchUseCase:
    """Use case for processing a batch of unprocessed issues."""

    def __init__(
        self,
        issue_repo: IssueRepository,
        preprocessed_repo: PreprocessedIssueRepository,
        preprocessor: TextPreprocessor,
        embedding_gen: EmbeddingGenerator,
        vectordb: VectorDBService,
    ):
        """
        Initialize use case with dependencies.

        Args:
            issue_repo: Issue repository
            preprocessed_repo: Preprocessed issue repository
            preprocessor: Text preprocessing service
            embedding_gen: Embedding generation service
            vectordb: Vector database service
        """
        self.issue_repo = issue_repo
        self.preprocessed_repo = preprocessed_repo
        self.preprocessor = preprocessor
        self.embedding_gen = embedding_gen
        self.vectordb = vectordb

    async def execute(
        self, batch_size: int = 100, device: str = "cpu"
    ) -> ProcessingResultDTO:
        """
        Process a batch of unprocessed issues through pipeline.

        Args:
            batch_size: Number of issues to process
            device: Device for embeddings (cpu or cuda)

        Returns:
            Processing result with statistics
        """
        logger.info(f"Processing batch of {batch_size} issues (device={device})")

        # Create configuration
        config = PipelineConfig(batch_size=batch_size, device=device)

        # Create pipeline stages
        stages = [
            Stage0FetchIssues(self.issue_repo),
            Stage1Preprocess(self.preprocessor, self.preprocessed_repo),
            Stage2GenerateEmbeddings(self.embedding_gen),
            Stage3VectorDBStorage(self.vectordb),
            Stage4Complete(),
        ]

        # Execute pipeline
        executor = PipelineExecutor(config, stages)
        result = await executor.execute()

        # Convert to DTO
        return ProcessingResultDTO(
            processed_count=result.processed_count,
            duration_seconds=result.duration_seconds,
            stats=result.stats,
            errors=result.errors,
            success=result.success,
        )
