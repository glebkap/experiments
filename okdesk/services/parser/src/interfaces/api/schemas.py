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


class IssueStatsResponse(BaseModel):
    """Statistics about issues."""

    total_issues: int = Field(..., description="Total number of issues")
    by_status: dict[str, int] = Field(
        default_factory=dict, description="Issue count by status"
    )
    by_priority: dict[str, int] = Field(
        default_factory=dict, description="Issue count by priority"
    )
    by_source: dict[str, int] = Field(
        default_factory=dict, description="Issue count by source"
    )


class MessageStatsResponse(BaseModel):
    """Statistics about messages."""

    total_messages: int = Field(..., description="Total number of messages")
    by_author_type: dict[str, int] = Field(
        default_factory=dict, description="Message count by author type"
    )
    public_messages: int = Field(..., description="Number of public messages")
    private_messages: int = Field(..., description="Number of private messages")


class ImportStatsResponse(BaseModel):
    """Statistics about imports."""

    total_imports: int = Field(..., description="Total number of imports")
    by_status: dict[str, int] = Field(
        default_factory=dict, description="Import count by status"
    )
    last_import: datetime | None = Field(None, description="Last import timestamp")


class GlobalStatsResponse(BaseModel):
    """Global statistics combining all entities."""

    issues: IssueStatsResponse
    messages: MessageStatsResponse
    imports: ImportStatsResponse
