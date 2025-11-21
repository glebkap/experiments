"""Pydantic schemas for API request/response."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ImportResponse(BaseModel):
    """Response after starting import."""

    import_id: UUID
    status: str
    message: str


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: str | None = None
