"""FastAPI routes for Parser Service."""

import logging
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from ...application.use_cases import ImportOKDeskUseCase
from ...config import settings
from ...infrastructure.persistence import get_db
from .schemas import ErrorResponse, ImportResponse

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
