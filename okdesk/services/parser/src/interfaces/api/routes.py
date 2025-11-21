"""FastAPI routes for Parser Service."""

import logging
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ...application.use_cases import ImportOKDeskUseCase
from ...config import settings
from ...infrastructure.persistence import get_db
from ...infrastructure.persistence.models import (
    ImportModel,
    IssueModel,
    MessageModel,
    SourceModel,
)
from .schemas import (
    ErrorResponse,
    GlobalStatsResponse,
    ImportResponse,
    ImportStatsResponse,
    IssueStatsResponse,
    MessageStatsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


@router.post("/import/okdesk", response_model=ImportResponse)
async def import_okdesk(
    file: UploadFile = File(...),
    source_id: UUID = None,
    db: AsyncSession = Depends(get_db),
):
    """Import OKDesk JSONL file."""
    if not source_id:
        # For MVP, create a default source_id
        source_id = UUID("00000000-0000-0000-0000-000000000001")

    # Save uploaded file
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / file.filename
    with file_path.open("wb") as f:
        content = await file.read()
        f.write(content)

    logger.info(f"Saved uploaded file to {file_path}")

    # Import file (use case will be injected via DI later)
    # For MVP, return success
    return ImportResponse(
        import_id=UUID("00000000-0000-0000-0000-000000000002"),
        status="in_progress",
        message=f"Import started for {file.filename}",
    )


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "parser"}


@router.get("/stats/issues", response_model=IssueStatsResponse)
async def get_issue_stats(db: AsyncSession = Depends(get_db)):
    """Get statistics about issues."""
    # Total issues
    total_result = await db.execute(select(func.count(IssueModel.id)))
    total_issues = total_result.scalar() or 0

    # By status
    status_result = await db.execute(
        select(IssueModel.status, func.count(IssueModel.id))
        .where(IssueModel.status.isnot(None))
        .group_by(IssueModel.status)
    )
    by_status = {str(status): count for status, count in status_result.all()}

    # By priority
    priority_result = await db.execute(
        select(IssueModel.priority, func.count(IssueModel.id))
        .where(IssueModel.priority.isnot(None))
        .group_by(IssueModel.priority)
    )
    by_priority = {str(priority): count for priority, count in priority_result.all()}

    # By source
    source_result = await db.execute(
        select(SourceModel.name, func.count(IssueModel.id))
        .join(IssueModel, IssueModel.source_id == SourceModel.id)
        .group_by(SourceModel.name)
    )
    by_source = {name: count for name, count in source_result.all()}

    return IssueStatsResponse(
        total_issues=total_issues,
        by_status=by_status,
        by_priority=by_priority,
        by_source=by_source,
    )


@router.get("/stats/messages", response_model=MessageStatsResponse)
async def get_message_stats(db: AsyncSession = Depends(get_db)):
    """Get statistics about messages."""
    # Total messages
    total_result = await db.execute(select(func.count(MessageModel.id)))
    total_messages = total_result.scalar() or 0

    # By author type
    author_type_result = await db.execute(
        select(MessageModel.author_type, func.count(MessageModel.id))
        .where(MessageModel.author_type.isnot(None))
        .group_by(MessageModel.author_type)
    )
    by_author_type = {
        str(author_type): count for author_type, count in author_type_result.all()
    }

    # Public vs private
    public_result = await db.execute(
        select(func.count(MessageModel.id)).where(MessageModel.is_public == True)
    )
    public_messages = public_result.scalar() or 0

    private_result = await db.execute(
        select(func.count(MessageModel.id)).where(MessageModel.is_public == False)
    )
    private_messages = private_result.scalar() or 0

    return MessageStatsResponse(
        total_messages=total_messages,
        by_author_type=by_author_type,
        public_messages=public_messages,
        private_messages=private_messages,
    )


@router.get("/stats/imports", response_model=ImportStatsResponse)
async def get_import_stats(db: AsyncSession = Depends(get_db)):
    """Get statistics about imports."""
    # Total imports
    total_result = await db.execute(select(func.count(ImportModel.id)))
    total_imports = total_result.scalar() or 0

    # By status
    status_result = await db.execute(
        select(ImportModel.status, func.count(ImportModel.id)).group_by(
            ImportModel.status
        )
    )
    by_status = {str(status): count for status, count in status_result.all()}

    # Last import
    last_import_result = await db.execute(
        select(ImportModel.started_at)
        .order_by(ImportModel.started_at.desc())
        .limit(1)
    )
    last_import = last_import_result.scalar_one_or_none()

    return ImportStatsResponse(
        total_imports=total_imports, by_status=by_status, last_import=last_import
    )


@router.get("/stats", response_model=GlobalStatsResponse)
async def get_global_stats(db: AsyncSession = Depends(get_db)):
    """Get global statistics combining all entities."""
    issues = await get_issue_stats(db)
    messages = await get_message_stats(db)
    imports = await get_import_stats(db)

    return GlobalStatsResponse(issues=issues, messages=messages, imports=imports)
