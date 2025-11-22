"""Pipeline context for passing data between stages."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ...domain.models.issue import Issue
from ...domain.models.preprocessed_issue import PreprocessedIssue
from .pipeline_config import PipelineConfig


@dataclass
class PipelineContext:
    """
    Shared context for pipeline execution.

    Contains data and statistics that are passed through all stages.
    """

    config: PipelineConfig
    issues: List[Issue] = field(default_factory=list)
    preprocessed_issues: List[PreprocessedIssue] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None
    stats: Dict[str, Any] = field(default_factory=dict)

    def add_stat(self, key: str, value: Any) -> None:
        """
        Add or update a statistic.

        Args:
            key: Statistic name
            value: Statistic value
        """
        self.stats[key] = value

    def increment_stat(self, key: str, increment: int = 1) -> None:
        """
        Increment a numeric statistic.

        Args:
            key: Statistic name
            increment: Amount to increment by
        """
        self.stats[key] = self.stats.get(key, 0) + increment

    def get_stat(self, key: str, default: Any = None) -> Any:
        """
        Get statistic value.

        Args:
            key: Statistic name
            default: Default value if not found

        Returns:
            Statistic value or default
        """
        return self.stats.get(key, default)

    def clear_issues(self) -> None:
        """Clear issues to free memory after processing."""
        self.issues = []

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of pipeline execution.

        Returns:
            Dictionary with key statistics
        """
        return {
            "issues_fetched": len(self.issues),
            "issues_preprocessed": len(self.preprocessed_issues),
            "embeddings_generated": self.embeddings.shape[0] if self.embeddings is not None else 0,
            "stats": self.stats,
        }
