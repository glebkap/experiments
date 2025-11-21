#!/usr/bin/env python3
"""Script for importing OKDesk JSONL files."""

import asyncio
import sys
from pathlib import Path
from uuid import uuid4

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.application.use_cases import ImportOKDeskUseCase
from src.domain.repositories import (
    ImportRepository,
    IssueRepository,
    MessageRepository,
    SourceRepository,
)
from src.domain.services import DeduplicationService, ImportService
from src.infrastructure.http import AnalyzerClient
from src.infrastructure.parsers import OKDeskParser
from src.infrastructure.persistence import AsyncSessionLocal
from src.infrastructure.persistence.postgres import (
    ImportRepositoryImpl,
    IssueRepositoryImpl,
    MessageRepositoryImpl,
    SourceRepositoryImpl,
)


async def main():
    """Run OKDesk import."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/import_okdesk.py <path_to_jsonl_file> [source_id]")
        print("\nExample:")
        print("  python scripts/import_okdesk.py ../../data/okdesk/2025-11-14.jsonl")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    # Use provided source_id or generate new one
    if len(sys.argv) > 2:
        from uuid import UUID

        source_id = UUID(sys.argv[2])
    else:
        # Generate default source_id for OKDesk
        source_id = uuid4()
        print(f"Generated source_id: {source_id}")

    print(f"Importing file: {file_path}")
    print(f"Source ID: {source_id}")

    # Create session
    async with AsyncSessionLocal() as session:
        try:
            # Create repositories
            import_repo: ImportRepository = ImportRepositoryImpl(session)
            issue_repo: IssueRepository = IssueRepositoryImpl(session)
            message_repo: MessageRepository = MessageRepositoryImpl(session)
            source_repo: SourceRepository = SourceRepositoryImpl(session)

            # Create services
            dedup_service = DeduplicationService(issue_repo, message_repo)
            import_service = ImportService(
                import_repo, issue_repo, message_repo, dedup_service
            )

            # Create parser and analyzer client
            parser = OKDeskParser()
            analyzer = AnalyzerClient("http://localhost:8002")  # Analyzer service URL

            # Create use case
            use_case = ImportOKDeskUseCase(import_service, parser, analyzer)

            # Execute import
            print("\n" + "=" * 60)
            print("Starting import...")
            print("=" * 60 + "\n")

            result = await use_case.execute(file_path, source_id)

            print("\n" + "=" * 60)
            print("Import completed!")
            print("=" * 60)
            print(f"\nStatus: {result.status}")
            print(f"Message: {result.message}")
            print(f"Import ID: {result.import_id}")

            if result.stats:
                print("\nStatistics:")
                for key, value in result.stats.items():
                    print(f"  {key}: {value}")

            await session.commit()

        except Exception as e:
            await session.rollback()
            print(f"\n❌ Import failed: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
