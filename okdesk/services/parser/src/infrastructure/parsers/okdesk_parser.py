"""OKDesk JSONL file parser."""

import json
import logging
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class OKDeskParser:
    """Parser for OKDesk JSONL files."""

    def parse_file(self, file_path: str | Path) -> Iterator[dict[str, Any]]:
        """
        Parse JSONL file line by line.

        Args:
            file_path: Path to JSONL file

        Yields:
            Dictionary for each issue with comments
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Parsing OKDesk file: {file_path}")
        line_num = 0

        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line_num += 1
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    yield data
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse line {line_num}: {e}")
                    continue

        logger.info(f"Parsed {line_num} lines from {file_path}")

    def extract_issue(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Extract issue information from JSON data.

        Args:
            data: Raw JSON data from JSONL line

        Returns:
            Dictionary with issue fields
        """
        # Try to get ID from different possible locations
        external_id = (
            data.get("issue_id")  # Top level issue_id
            or data.get("id")  # Direct id field
            or (data.get("details", {}).get("id") if isinstance(data.get("details"), dict) else None)  # details.id
        )

        # Get details dict for other fields
        details = data.get("details", {}) if isinstance(data.get("details"), dict) else {}

        return {
            "external_id": str(external_id) if external_id else "",
            "title": self._clean_html(details.get("title") or data.get("title")),
            "description": self._clean_html(details.get("description") or data.get("description")),
            "status": self._map_status(details.get("status") or data.get("status")),
            "priority": self._map_priority(details.get("priority") or data.get("priority")),
            "created_at": self._parse_datetime(details.get("created_at") or data.get("created_at")),
            "updated_at": self._parse_datetime(details.get("updated_at") or data.get("updated_at")),
            "completed_at": self._parse_datetime(details.get("completed_at") or data.get("completed_at")),
        }

    def extract_comments(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Extract comments/messages from issue data.

        Args:
            data: Raw JSON data from JSONL line

        Returns:
            List of comment dictionaries (skips messages with empty content)
        """
        comments = data.get("comments", [])
        if not isinstance(comments, list):
            return []

        result = []
        for comment in comments:
            if not isinstance(comment, dict):
                continue

            # Clean content
            content = self._clean_html(comment.get("content", ""))

            # Skip messages with empty content
            if not content:
                logger.debug(f"Skipping comment {comment.get('id')} with empty content")
                continue

            # Extract author information
            author = comment.get("author", {}) or {}
            author_type = self._map_author_type(author.get("type"))

            result.append(
                {
                    "external_id": str(comment.get("id", "")),
                    "author_id": str(author.get("id", "")) if author.get("id") else None,
                    "author_name": author.get("name"),
                    "author_type": author_type,
                    "content": content,
                    "is_public": comment.get("public", True),
                    "published_at": self._parse_datetime(comment.get("published_at")),
                }
            )

        return result

    def _clean_html(self, text: str | None) -> str | None:
        """
        Clean HTML tags from text using BeautifulSoup.

        Args:
            text: Text potentially containing HTML

        Returns:
            Plain text without HTML tags
        """
        if not text:
            return None

        # Parse HTML and extract text
        soup = BeautifulSoup(text, "lxml")
        clean_text = soup.get_text(separator=" ", strip=True)

        return clean_text if clean_text else None

    def _map_status(self, status: str | dict | None) -> str | None:
        """Map OKDesk status to our enum values."""
        if not status:
            return None

        # Handle dict format {"code": "...", "name": "..."}
        if isinstance(status, dict):
            status = status.get("code")
            if not status:
                return None

        status_map = {
            "opened": "opened",
            "open": "opened",
            "wait": "wait",
            "waiting": "wait",
            "completed": "completed",
            "complete": "completed",
            "closed": "closed",
            "close": "closed",
        }

        return status_map.get(status.lower(), None)

    def _map_priority(self, priority: Any) -> int | None:
        """Map priority to integer 1-4."""
        if priority is None:
            return None

        try:
            p = int(priority)
            return p if 1 <= p <= 4 else None
        except (ValueError, TypeError):
            return None

    def _map_author_type(self, author_type: str | None) -> str | None:
        """Map OKDesk author type to our enum."""
        if not author_type:
            return None

        type_map = {
            "employee": "employee",
            "contact": "contact",
            "user": "user",
            "client": "contact",
            "customer": "contact",
        }

        return type_map.get(author_type.lower(), None)

    def _parse_datetime(self, dt_str: str | None) -> datetime | None:
        """
        Parse datetime string to datetime object.

        Supports various formats from OKDesk API including timezone offsets.
        Returns naive datetime in UTC.
        """
        if not dt_str:
            return None

        # Try datetime.fromisoformat first (handles timezone offsets like +03:00)
        try:
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            # Convert to UTC and remove timezone info for PostgreSQL
            if dt.tzinfo is not None:
                dt = dt.astimezone(None).replace(tzinfo=None)
            return dt
        except (ValueError, AttributeError):
            pass

        # Try different datetime formats
        formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",  # ISO format with microseconds
            "%Y-%m-%dT%H:%M:%SZ",  # ISO format without microseconds
            "%Y-%m-%d %H:%M:%S",  # Standard format
            "%Y-%m-%d",  # Date only
        ]

        for fmt in formats:
            try:
                return datetime.strptime(dt_str, fmt)
            except ValueError:
                continue

        logger.warning(f"Could not parse datetime: {dt_str}")
        return None
