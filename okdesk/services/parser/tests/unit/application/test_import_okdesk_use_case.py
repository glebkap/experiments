"""Unit tests for ImportOKDeskUseCase."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from uuid import uuid4
from datetime import datetime

from src.application.use_cases import ImportOKDeskUseCase
from src.application.dto import ImportResponse
from src.domain.models import ImportJob, ImportStatus, Issue, Message


class TestImportOKDeskUseCase:
    """Tests for ImportOKDeskUseCase."""

    @pytest.mark.asyncio
    async def test_file_not_found_raises_error(self):
        """Test that non-existent file raises FileNotFoundError."""
        import_service = AsyncMock()
        parser = MagicMock()
        analyzer = AsyncMock()

        use_case = ImportOKDeskUseCase(import_service, parser, analyzer)

        with pytest.raises(FileNotFoundError):
            await use_case.execute("/nonexistent/file.jsonl", uuid4())

    @pytest.mark.asyncio
    async def test_creates_import_job_with_in_progress_status(self):
        """Test that import job is created with IN_PROGRESS status."""
        import_service = AsyncMock()
        parser = MagicMock()
        analyzer = AsyncMock()

        # Mock file existence
        with patch("pathlib.Path.exists", return_value=True):
            # Mock parser to return no data
            parser.parse_file.return_value = iter([])

            # Mock import_service
            mock_job = ImportJob(
                id=uuid4(),
                source_id=uuid4(),
                filename="test.jsonl",
                file_path="/test.jsonl",
                started_at=datetime.utcnow(),
                completed_at=None,
                status=ImportStatus.IN_PROGRESS,
                stats=None,
                error_message=None,
            )
            import_service.create_import_job.return_value = mock_job

            use_case = ImportOKDeskUseCase(import_service, parser, analyzer)

            await use_case.execute(Path("/test.jsonl"), uuid4(), "test.jsonl")

            # Verify import job was created
            import_service.create_import_job.assert_called_once()
            created_job = import_service.create_import_job.call_args[0][0]
            assert created_job.status == ImportStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_uses_filename_from_path_if_not_provided(self):
        """Test that filename is extracted from path if not provided."""
        import_service = AsyncMock()
        parser = MagicMock()
        analyzer = AsyncMock()

        with patch("pathlib.Path.exists", return_value=True):
            parser.parse_file.return_value = iter([])

            mock_job = ImportJob(
                id=uuid4(),
                source_id=uuid4(),
                filename="data.jsonl",
                file_path="/path/to/data.jsonl",
                started_at=datetime.utcnow(),
                completed_at=None,
                status=ImportStatus.IN_PROGRESS,
                stats=None,
                error_message=None,
            )
            import_service.create_import_job.return_value = mock_job

            use_case = ImportOKDeskUseCase(import_service, parser, analyzer)

            file_path = Path("/path/to/data.jsonl")
            await use_case.execute(file_path, uuid4())

            # Check that filename was extracted from path
            created_job = import_service.create_import_job.call_args[0][0]
            assert created_job.filename == "data.jsonl"

    @pytest.mark.asyncio
    async def test_marks_job_completed_on_success(self):
        """Test that job is marked as completed after successful import."""
        import_service = AsyncMock()
        parser = MagicMock()
        analyzer = AsyncMock()

        with patch("pathlib.Path.exists", return_value=True):
            # Mock parser with one issue and one comment
            parser.parse_file.return_value = iter([
                {
                    "id": 123,
                    "title": "Test",
                    "comments": [{"id": 1, "content": "Comment"}]
                }
            ])
            parser.extract_issue.return_value = {
                "external_id": "123",
                "title": "Test",
                "description": None,
                "status": "opened",
                "priority": None,
                "created_at": None,
                "updated_at": None,
                "completed_at": None,
            }
            parser.extract_comments.return_value = [
                {
                    "external_id": "1",
                    "author_id": None,
                    "author_name": None,
                    "author_type": None,
                    "content": "Comment",
                    "is_public": True,
                    "published_at": None,
                }
            ]

            # Mock services
            mock_job = ImportJob(
                id=uuid4(),
                source_id=uuid4(),
                filename="test.jsonl",
                file_path="/test.jsonl",
                started_at=datetime.utcnow(),
                completed_at=None,
                status=ImportStatus.IN_PROGRESS,
                stats=None,
                error_message=None,
            )
            import_service.create_import_job.return_value = mock_job

            mock_issue = Issue(
                id=uuid4(),
                external_id="123",
                source_id=uuid4(),
                title="Test",
                description=None,
                status="opened",
                priority=None,
                created_at=None,
                updated_at=None,
                completed_at=None,
            )
            import_service.process_issue.return_value = (mock_issue, True)

            mock_message = Message(
                id=uuid4(),
                issue_id=mock_issue.id,
                external_id="1",
                author_id=None,
                author_name=None,
                author_type=None,
                content="Comment",
                is_public=True,
                published_at=None,
            )
            import_service.process_message.return_value = (mock_message, True)

            import_service.calculate_stats.return_value = {
                "total_issues": 1,
                "new_issues": 1,
                "total_messages": 1,
                "new_messages": 1,
            }

            use_case = ImportOKDeskUseCase(import_service, parser, analyzer)

            result = await use_case.execute(Path("/test.jsonl"), uuid4())

            # Verify job was marked completed
            import_service.update_import_job.assert_called()
            assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_marks_job_failed_on_exception(self):
        """Test that job is marked as failed if exception occurs."""
        import_service = AsyncMock()
        parser = MagicMock()
        analyzer = AsyncMock()

        with patch("pathlib.Path.exists", return_value=True):
            # Make parser raise exception
            parser.parse_file.side_effect = Exception("Parser error")

            mock_job = ImportJob(
                id=uuid4(),
                source_id=uuid4(),
                filename="test.jsonl",
                file_path="/test.jsonl",
                started_at=datetime.utcnow(),
                completed_at=None,
                status=ImportStatus.IN_PROGRESS,
                stats=None,
                error_message=None,
            )
            import_service.create_import_job.return_value = mock_job

            use_case = ImportOKDeskUseCase(import_service, parser, analyzer)

            result = await use_case.execute(Path("/test.jsonl"), uuid4())

            # Verify job was marked failed
            import_service.update_import_job.assert_called()
            assert result.status == "failed"
            assert "Parser error" in result.message

    @pytest.mark.asyncio
    async def test_triggers_analyzer_with_new_message_ids(self):
        """Test that analyzer is triggered with new message IDs."""
        import_service = AsyncMock()
        parser = MagicMock()
        analyzer = AsyncMock()

        with patch("pathlib.Path.exists", return_value=True):
            parser.parse_file.return_value = iter([
                {"id": 1, "comments": [{"id": 10, "content": "Msg1"}]}
            ])
            parser.extract_issue.return_value = {
                "external_id": "1",
                "title": None,
                "description": None,
                "status": None,
                "priority": None,
                "created_at": None,
                "updated_at": None,
                "completed_at": None,
            }
            parser.extract_comments.return_value = [
                {
                    "external_id": "10",
                    "author_id": None,
                    "author_name": None,
                    "author_type": None,
                    "content": "Msg1",
                    "is_public": True,
                    "published_at": None,
                }
            ]

            mock_job = ImportJob(
                id=uuid4(),
                source_id=uuid4(),
                filename="test.jsonl",
                file_path="/test.jsonl",
                started_at=datetime.utcnow(),
                completed_at=None,
                status=ImportStatus.IN_PROGRESS,
                stats=None,
                error_message=None,
            )
            import_service.create_import_job.return_value = mock_job

            issue_id = uuid4()
            mock_issue = Issue(
                id=issue_id,
                external_id="1",
                source_id=uuid4(),
                title=None,
                description=None,
                status=None,
                priority=None,
                created_at=None,
                updated_at=None,
                completed_at=None,
            )
            import_service.process_issue.return_value = (mock_issue, True)

            message_id = uuid4()
            mock_message = Message(
                id=message_id,
                issue_id=issue_id,
                external_id="10",
                author_id=None,
                author_name=None,
                author_type=None,
                content="Msg1",
                is_public=True,
                published_at=None,
            )
            import_service.process_message.return_value = (mock_message, True)  # is_new=True

            import_service.calculate_stats.return_value = {}

            use_case = ImportOKDeskUseCase(import_service, parser, analyzer)

            await use_case.execute(Path("/test.jsonl"), uuid4())

            # Verify analyzer was called with the new message ID
            analyzer.analyze_batch.assert_called_once()
            call_args = analyzer.analyze_batch.call_args[0]
            assert message_id in call_args[0]

    @pytest.mark.asyncio
    async def test_does_not_trigger_analyzer_if_no_new_messages(self):
        """Test that analyzer is not triggered if no new messages."""
        import_service = AsyncMock()
        parser = MagicMock()
        analyzer = AsyncMock()

        with patch("pathlib.Path.exists", return_value=True):
            parser.parse_file.return_value = iter([])

            mock_job = ImportJob(
                id=uuid4(),
                source_id=uuid4(),
                filename="test.jsonl",
                file_path="/test.jsonl",
                started_at=datetime.utcnow(),
                completed_at=None,
                status=ImportStatus.IN_PROGRESS,
                stats=None,
                error_message=None,
            )
            import_service.create_import_job.return_value = mock_job
            import_service.calculate_stats.return_value = {}

            use_case = ImportOKDeskUseCase(import_service, parser, analyzer)

            await use_case.execute(Path("/test.jsonl"), uuid4())

            # Verify analyzer was NOT called
            analyzer.analyze_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_continues_on_analyzer_failure(self):
        """Test that import completes even if analyzer fails."""
        import_service = AsyncMock()
        parser = MagicMock()
        analyzer = AsyncMock()

        with patch("pathlib.Path.exists", return_value=True):
            parser.parse_file.return_value = iter([])

            mock_job = ImportJob(
                id=uuid4(),
                source_id=uuid4(),
                filename="test.jsonl",
                file_path="/test.jsonl",
                started_at=datetime.utcnow(),
                completed_at=None,
                status=ImportStatus.IN_PROGRESS,
                stats=None,
                error_message=None,
            )
            import_service.create_import_job.return_value = mock_job
            import_service.calculate_stats.return_value = {}

            # Make analyzer fail
            analyzer.analyze_batch.side_effect = Exception("Analyzer down")

            use_case = ImportOKDeskUseCase(import_service, parser, analyzer)

            result = await use_case.execute(Path("/test.jsonl"), uuid4())

            # Import should still be completed
            assert result.status == "completed"
