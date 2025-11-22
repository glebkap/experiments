"""DTO for processing results."""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ProcessingResultDTO:
    """Data transfer object for pipeline processing results."""

    processed_count: int
    duration_seconds: float
    stats: Dict[str, Any]
    errors: List[str]
    success: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "processed_count": self.processed_count,
            "duration_seconds": round(self.duration_seconds, 2),
            "stats": self.stats,
            "errors": self.errors,
            "success": self.success,
        }
