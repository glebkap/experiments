#!/usr/bin/env python3
"""Script for displaying database statistics."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import func, select

from src.infrastructure.persistence import AsyncSessionLocal
from src.infrastructure.persistence.models import (
    ImportModel,
    IssueModel,
    MessageModel,
    SourceModel,
)


async def main():
    """Display database statistics."""
    async with AsyncSessionLocal() as session:
        print("\n" + "=" * 70)
        print("DATABASE STATISTICS")
        print("=" * 70 + "\n")

        # Sources
        print("📦 SOURCES")
        print("-" * 70)
        sources_result = await session.execute(
            select(SourceModel.name, SourceModel.type, func.count(IssueModel.id))
            .outerjoin(IssueModel, IssueModel.source_id == SourceModel.id)
            .group_by(SourceModel.id, SourceModel.name, SourceModel.type)
        )
        sources = sources_result.all()
        for name, source_type, count in sources:
            print(f"  • {name} ({source_type}): {count} issues")
        print()

        # Issues
        print("🎫 ISSUES")
        print("-" * 70)

        # Total
        total_issues_result = await session.execute(select(func.count(IssueModel.id)))
        total_issues = total_issues_result.scalar() or 0
        print(f"  Total: {total_issues}")

        # By status
        status_result = await session.execute(
            select(IssueModel.status, func.count(IssueModel.id))
            .where(IssueModel.status.isnot(None))
            .group_by(IssueModel.status)
            .order_by(func.count(IssueModel.id).desc())
        )
        print("\n  By Status:")
        for status, count in status_result.all():
            percentage = (count / total_issues * 100) if total_issues > 0 else 0
            print(f"    • {status}: {count} ({percentage:.1f}%)")

        # By priority
        priority_result = await session.execute(
            select(IssueModel.priority, func.count(IssueModel.id))
            .where(IssueModel.priority.isnot(None))
            .group_by(IssueModel.priority)
            .order_by(IssueModel.priority)
        )
        print("\n  By Priority:")
        for priority, count in priority_result.all():
            percentage = (count / total_issues * 100) if total_issues > 0 else 0
            print(f"    • Priority {priority}: {count} ({percentage:.1f}%)")
        print()

        # Messages
        print("💬 MESSAGES")
        print("-" * 70)

        # Total
        total_messages_result = await session.execute(
            select(func.count(MessageModel.id))
        )
        total_messages = total_messages_result.scalar() or 0
        print(f"  Total: {total_messages}")

        # By author type
        author_type_result = await session.execute(
            select(MessageModel.author_type, func.count(MessageModel.id))
            .where(MessageModel.author_type.isnot(None))
            .group_by(MessageModel.author_type)
            .order_by(func.count(MessageModel.id).desc())
        )
        print("\n  By Author Type:")
        for author_type, count in author_type_result.all():
            percentage = (count / total_messages * 100) if total_messages > 0 else 0
            print(f"    • {author_type}: {count} ({percentage:.1f}%)")

        # Public vs Private
        public_result = await session.execute(
            select(func.count(MessageModel.id)).where(MessageModel.is_public == True)
        )
        public_count = public_result.scalar() or 0

        private_result = await session.execute(
            select(func.count(MessageModel.id)).where(MessageModel.is_public == False)
        )
        private_count = private_result.scalar() or 0

        print("\n  By Visibility:")
        if total_messages > 0:
            print(f"    • Public: {public_count} ({public_count / total_messages * 100:.1f}%)")
            print(f"    • Private: {private_count} ({private_count / total_messages * 100:.1f}%)")
        else:
            print(f"    • Public: {public_count}")
            print(f"    • Private: {private_count}")
        print()

        # Imports
        print("📥 IMPORTS")
        print("-" * 70)

        # Total
        total_imports_result = await session.execute(
            select(func.count(ImportModel.id))
        )
        total_imports = total_imports_result.scalar() or 0
        print(f"  Total: {total_imports}")

        # By status
        import_status_result = await session.execute(
            select(ImportModel.status, func.count(ImportModel.id))
            .group_by(ImportModel.status)
            .order_by(func.count(ImportModel.id).desc())
        )
        print("\n  By Status:")
        for status, count in import_status_result.all():
            percentage = (count / total_imports * 100) if total_imports > 0 else 0
            print(f"    • {status}: {count} ({percentage:.1f}%)")

        # Last import
        last_import_result = await session.execute(
            select(
                ImportModel.started_at,
                ImportModel.completed_at,
                ImportModel.status,
                ImportModel.filename,
            )
            .order_by(ImportModel.started_at.desc())
            .limit(1)
        )
        last_import = last_import_result.first()
        if last_import:
            started, completed, status, filename = last_import
            print("\n  Last Import:")
            print(f"    • File: {filename}")
            print(f"    • Status: {status}")
            print(f"    • Started: {started}")
            if completed:
                print(f"    • Completed: {completed}")
                duration = (completed - started).total_seconds()
                print(f"    • Duration: {duration:.1f}s")

        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
