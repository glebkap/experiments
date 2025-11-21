"""Use case for importing OKDesk data."""

import logging
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

from ...domain.models import ImportJob, ImportStatus, Issue, Message
from ...domain.services import ImportService
from ...infrastructure.http import AnalyzerClient
from ...infrastructure.parsers import OKDeskParser
from ..dto import ImportResponse

logger = logging.getLogger(__name__)


class ImportOKDeskUseCase:
    """Use case for importing OKDesk JSONL files."""

    def __init__(
        self,
        import_service: ImportService,
        parser: OKDeskParser,
        analyzer_client: AnalyzerClient,
        session=None,
    ) -> None:
        """Initialize use case with dependencies."""
        self._import_service = import_service
        self._parser = parser
        self._analyzer = analyzer_client
        self._session = session

    async def execute(
        self, file_path: str | Path, source_id: UUID, filename: str | None = None
    ) -> ImportResponse:
        """
        Execute OKDesk import.

        Args:
            file_path: Path to JSONL file
            source_id: Source identifier
            filename: Optional custom filename

        Returns:
            ImportResponse with import_id and status
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Create import job
        import_job = ImportJob(
            id=uuid4(),
            source_id=source_id,
            filename=filename or file_path.name,
            file_path=str(file_path),
            started_at=datetime.utcnow(),
            completed_at=None,
            status=ImportStatus.IN_PROGRESS,
            stats=None,
            error_message=None,
        )

        import_job = await self._import_service.create_import_job(import_job)
        logger.info(f"Created import job {import_job.id}")

        try:
            # Count total lines in file for progress percentage
            logger.info("Counting total records in file...")
            total_lines = 0
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        total_lines += 1
            logger.info(f"Total records to process: {total_lines}")

            # Process file
            total_issues = 0
            new_issues = 0
            updated_issues = 0
            unchanged_issues = 0
            total_messages = 0
            new_messages = 0
            updated_messages = 0
            unchanged_messages = 0
            skipped_issues_empty_desc = 0
            skipped_messages_empty_content = 0
            message_ids: list[UUID] = []
            processed_lines = 0

            logger.info("Starting import processing...")

            for data in self._parser.parse_file(file_path):
                processed_lines += 1
                # Extract and process issue
                issue_data = self._parser.extract_issue(data)

                # Skip issues with empty description
                if not issue_data["description"]:
                    logger.debug(f"Skipping issue {issue_data['external_id']} with empty description")
                    skipped_issues_empty_desc += 1
                    continue

                issue = Issue(
                    id=uuid4(),
                    external_id=issue_data["external_id"],
                    source_id=source_id,
                    title=issue_data["title"],
                    description=issue_data["description"],
                    status=issue_data["status"],
                    priority=issue_data["priority"],
                    created_at=issue_data["created_at"],
                    updated_at=issue_data["updated_at"],
                    completed_at=issue_data["completed_at"],
                )

                processed_issue, status = await self._import_service.process_issue(issue)
                total_issues += 1
                if status == 'created':
                    new_issues += 1
                elif status == 'updated':
                    updated_issues += 1
                elif status == 'unchanged':
                    unchanged_issues += 1

                # Extract and process comments
                comments = self._parser.extract_comments(data)
                for comment_data in comments:
                    # Skip messages with empty content
                    if not comment_data["content"]:
                        logger.debug(f"Skipping message {comment_data['external_id']} with empty content")
                        skipped_messages_empty_content += 1
                        continue

                    message = Message(
                        id=uuid4(),
                        issue_id=processed_issue.id,
                        external_id=comment_data["external_id"],
                        author_id=comment_data["author_id"],
                        author_name=comment_data["author_name"],
                        author_type=comment_data["author_type"],
                        content=comment_data["content"],
                        is_public=comment_data["is_public"],
                        published_at=comment_data["published_at"],
                    )

                    processed_msg, msg_status = await self._import_service.process_message(
                        message
                    )
                    total_messages += 1
                    if msg_status == 'created':
                        new_messages += 1
                        message_ids.append(processed_msg.id)
                    elif msg_status == 'updated':
                        updated_messages += 1
                    elif msg_status == 'unchanged':
                        unchanged_messages += 1

                # Commit each record to make it visible immediately in statistics
                if self._session:
                    await self._session.commit()

                # Log progress every 100 issues
                if total_issues % 100 == 0:
                    progress_pct = (processed_lines / total_lines * 100) if total_lines > 0 else 0
                    logger.info(
                        f"Progress: {progress_pct:.1f}% ({processed_lines}/{total_lines}) | "
                        f"Issues: {total_issues} ({new_issues} new, {updated_issues} updated, {unchanged_issues} unchanged) | "
                        f"Messages: {total_messages} ({new_messages} new, {updated_messages} updated, {unchanged_messages} unchanged) | "
                        f"Skipped: {skipped_issues_empty_desc} issues, {skipped_messages_empty_content} messages"
                    )

            # Calculate stats
            stats = self._import_service.calculate_stats(
                total_issues, new_issues, total_messages, new_messages
            )

            # Log final summary
            logger.info("=" * 60)
            logger.info("Import completed successfully!")
            logger.info(f"Total issues processed: {total_issues}")
            logger.info(f"  - Created: {new_issues}")
            logger.info(f"  - Updated: {updated_issues}")
            logger.info(f"  - Unchanged: {unchanged_issues}")
            logger.info(f"Total messages processed: {total_messages}")
            logger.info(f"  - Created: {new_messages}")
            logger.info(f"  - Updated: {updated_messages}")
            logger.info(f"  - Unchanged: {unchanged_messages}")
            logger.info(f"Skipped:")
            logger.info(f"  - Issues (empty description): {skipped_issues_empty_desc}")
            logger.info(f"  - Messages (empty content): {skipped_messages_empty_content}")
            logger.info("=" * 60)

            # Mark as completed
            import_job.mark_completed(stats)
            await self._import_service.update_import_job(import_job)

            # Trigger analysis (fire and forget, don't wait)
            if message_ids:
                logger.info(f"Triggering analysis for {len(message_ids)} messages")
                try:
                    await self._analyzer.analyze_batch(message_ids, batch_size=10)
                except Exception as e:
                    logger.warning(f"Failed to trigger analysis: {e}")

            return ImportResponse(
                import_id=import_job.id,
                status="completed",
                message=f"Imported {new_issues} new issues, {new_messages} new messages",
            )

        except Exception as e:
            logger.error(f"Import failed: {e}", exc_info=True)
            import_job.mark_failed(str(e))
            await self._import_service.update_import_job(import_job)

            return ImportResponse(
                import_id=import_job.id, status="failed", message=f"Import failed: {e}"
            )
