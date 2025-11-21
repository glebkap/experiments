"""Telegram JSON export parser."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TelegramParser:
    """Parser for Telegram JSON export files."""

    def parse_file(self, file_path: str | Path) -> dict[str, Any]:
        """
        Parse Telegram export JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            Dictionary with parsed data
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Parsing Telegram file: {file_path}")

        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    def extract_messages(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Extract messages from Telegram export.

        Args:
            data: Parsed JSON data

        Returns:
            List of message dictionaries
        """
        messages = data.get("messages", [])
        if not isinstance(messages, list):
            return []

        result = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue

            # Skip service messages
            if msg.get("type") == "service":
                continue

            # Extract message text
            content = self._extract_text(msg)
            if not content:
                continue

            # Extract sender info
            from_user = msg.get("from", "")
            from_id = msg.get("from_id", "")

            result.append(
                {
                    "external_id": str(msg.get("id", "")),
                    "author_id": self._extract_user_id(from_id),
                    "author_name": from_user,
                    "author_type": self._determine_author_type(from_user, from_id),
                    "content": content,
                    "is_public": True,
                    "published_at": self._parse_datetime(msg.get("date")),
                }
            )

        logger.info(f"Extracted {len(result)} messages from Telegram export")
        return result

    def _extract_text(self, message: dict[str, Any]) -> str | None:
        """
        Extract text content from message.

        Telegram messages can have text as string or array of objects.
        """
        text = message.get("text", "")

        if isinstance(text, str):
            return text if text else None

        if isinstance(text, list):
            # Concatenate all text parts
            parts = []
            for item in text:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(item.get("text", ""))

            result = "".join(parts)
            return result if result else None

        return None

    def _extract_user_id(self, from_id: Any) -> str | None:
        """
        Extract user ID from various formats.

        Telegram can provide user_id as string or dict like {'user_id': 123}.
        """
        if not from_id:
            return None

        if isinstance(from_id, str):
            return from_id

        if isinstance(from_id, (int, float)):
            return str(from_id)

        if isinstance(from_id, dict):
            user_id = from_id.get("user_id")
            return str(user_id) if user_id else None

        return None

    def _determine_author_type(self, from_user: str | None, from_id: Any) -> str:
        """
        Determine author type based on available information.

        For Telegram, we can only distinguish between known users and contacts.
        """
        # Simple heuristic: if we have full info, it's a user; otherwise, contact
        if from_user and from_id:
            return "user"
        return "contact"

    def _parse_datetime(self, dt_str: str | None) -> datetime | None:
        """
        Parse datetime string from Telegram export.

        Args:
            dt_str: Datetime string from Telegram

        Returns:
            datetime object or None
        """
        if not dt_str:
            return None

        # Telegram export format: "2024-11-13T10:30:45"
        formats = [
            "%Y-%m-%dT%H:%M:%S",  # Standard Telegram format
            "%Y-%m-%d %H:%M:%S",  # Alternative format
            "%Y-%m-%d",  # Date only
        ]

        for fmt in formats:
            try:
                return datetime.strptime(dt_str, fmt)
            except ValueError:
                continue

        logger.warning(f"Could not parse Telegram datetime: {dt_str}")
        return None

    def create_synthetic_issue(
        self, chat_name: str | None, chat_id: Any
    ) -> dict[str, Any]:
        """
        Create a synthetic issue for Telegram chat.

        Since Telegram doesn't have "issues", we create a synthetic one
        representing the entire chat conversation.

        Args:
            chat_name: Name of the chat
            chat_id: Chat identifier

        Returns:
            Dictionary with synthetic issue data
        """
        return {
            "external_id": f"telegram_chat_{chat_id}",
            "title": chat_name or f"Telegram Chat {chat_id}",
            "description": "Messages from Telegram export",
            "status": "completed",
            "priority": None,
            "created_at": None,
            "updated_at": None,
            "completed_at": None,
        }
