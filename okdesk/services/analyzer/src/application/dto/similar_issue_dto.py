"""DTO for similar issues search results."""

from dataclasses import dataclass


@dataclass
class SimilarIssueDTO:
    """Similar issue found through semantic search."""

    issue_id: str
    title: str
    description: str
    similarity_score: float

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "issue_id": self.issue_id,
            "title": self.title,
            "description": self.description[:200] + "..." if len(self.description) > 200 else self.description,
            "similarity_score": round(self.similarity_score, 4),
        }
