#!/usr/bin/env python3
"""Script for clearing all database tables."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text

from src.infrastructure.persistence import AsyncSessionLocal


async def main():
    """Clear all database tables."""
    print("\n" + "=" * 70)
    print("⚠️  DATABASE CLEANUP - THIS WILL DELETE ALL DATA")
    print("=" * 70 + "\n")

    # Ask for confirmation
    confirmation = input("Are you sure you want to delete ALL data? Type 'yes' to confirm: ")
    if confirmation.lower() != "yes":
        print("Cleanup cancelled.")
        return

    async with AsyncSessionLocal() as session:
        try:
            print("\nDeleting data from tables...")

            # Delete in correct order (respecting foreign key constraints)
            tables = [
                "message_intents",
                "message_tags",
                "message_clusters",
                "message_analysis",
                "messages",
                "issues",
                "imports",
                "tags",
                "intent_clusters",
                "intents",
                "sources",
            ]

            for table in tables:
                result = await session.execute(text(f"DELETE FROM {table}"))
                deleted = result.rowcount
                print(f"  ✓ Deleted {deleted} rows from {table}")

            await session.commit()

            print("\n" + "=" * 70)
            print("✓ Database cleanup completed successfully!")
            print("=" * 70 + "\n")

        except Exception as e:
            await session.rollback()
            print(f"\n❌ Cleanup failed: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
