"""Unit tests for Telegram parser."""

import pytest
from datetime import datetime
from pathlib import Path
from src.infrastructure.parsers import TelegramParser


class TestTelegramParser:
    """Tests for TelegramParser."""

    def test_extract_text_from_string(self):
        """Test extracting text when it's a simple string."""
        parser = TelegramParser()

        message = {"text": "Hello world"}
        result = parser._extract_text(message)
        assert result == "Hello world"

    def test_extract_text_from_array(self):
        """Test extracting text when it's an array of objects."""
        parser = TelegramParser()

        message = {
            "text": [
                "Hello ",
                {"text": "world", "type": "bold"},
                "!"
            ]
        }
        result = parser._extract_text(message)
        assert result == "Hello world!"

    def test_extract_text_empty(self):
        """Test extracting text when empty."""
        parser = TelegramParser()

        message = {"text": ""}
        result = parser._extract_text(message)
        assert result is None

    def test_extract_text_missing(self):
        """Test extracting text when field is missing."""
        parser = TelegramParser()

        message = {}
        result = parser._extract_text(message)
        assert result is None

    def test_extract_user_id_from_string(self):
        """Test extracting user ID from string."""
        parser = TelegramParser()

        result = parser._extract_user_id("user123")
        assert result == "user123"

    def test_extract_user_id_from_int(self):
        """Test extracting user ID from integer."""
        parser = TelegramParser()

        result = parser._extract_user_id(12345)
        assert result == "12345"

    def test_extract_user_id_from_dict(self):
        """Test extracting user ID from dict."""
        parser = TelegramParser()

        result = parser._extract_user_id({"user_id": 999})
        assert result == "999"

    def test_extract_user_id_none(self):
        """Test extracting user ID when None."""
        parser = TelegramParser()

        result = parser._extract_user_id(None)
        assert result is None

    def test_determine_author_type_with_full_info(self):
        """Test author type determination with full info."""
        parser = TelegramParser()

        result = parser._determine_author_type("John Doe", "user123")
        assert result == "user"

    def test_determine_author_type_without_info(self):
        """Test author type determination without info."""
        parser = TelegramParser()

        result = parser._determine_author_type(None, None)
        assert result == "contact"

    def test_parse_datetime_standard_format(self):
        """Test parsing standard Telegram datetime."""
        parser = TelegramParser()

        dt = parser._parse_datetime("2024-11-21T10:30:45")
        assert isinstance(dt, datetime)
        assert dt.year == 2024
        assert dt.month == 11
        assert dt.day == 21

    def test_parse_datetime_alternative_format(self):
        """Test parsing alternative datetime format."""
        parser = TelegramParser()

        dt = parser._parse_datetime("2024-11-21 10:30:45")
        assert isinstance(dt, datetime)

    def test_parse_datetime_date_only(self):
        """Test parsing date only."""
        parser = TelegramParser()

        dt = parser._parse_datetime("2024-11-21")
        assert isinstance(dt, datetime)
        assert dt.year == 2024

    def test_parse_datetime_invalid(self):
        """Test parsing invalid datetime."""
        parser = TelegramParser()

        result = parser._parse_datetime("invalid-date")
        assert result is None

    def test_parse_datetime_none(self):
        """Test parsing None datetime."""
        parser = TelegramParser()

        result = parser._parse_datetime(None)
        assert result is None

    def test_create_synthetic_issue(self):
        """Test creating synthetic issue for Telegram chat."""
        parser = TelegramParser()

        issue = parser.create_synthetic_issue("Support Chat", 12345)

        assert issue["external_id"] == "telegram_chat_12345"
        assert issue["title"] == "Support Chat"
        assert issue["description"] == "Messages from Telegram export"
        assert issue["status"] == "completed"
        assert issue["priority"] is None

    def test_create_synthetic_issue_no_name(self):
        """Test creating synthetic issue without chat name."""
        parser = TelegramParser()

        issue = parser.create_synthetic_issue(None, 999)

        assert issue["external_id"] == "telegram_chat_999"
        assert "Telegram Chat 999" in issue["title"]

    def test_extract_messages_filters_service_messages(self):
        """Test that service messages are filtered out."""
        parser = TelegramParser()

        data = {
            "messages": [
                {
                    "id": 1,
                    "type": "message",
                    "text": "Normal message",
                    "from": "John",
                    "date": "2024-11-21T10:00:00"
                },
                {
                    "id": 2,
                    "type": "service",
                    "text": "User joined",
                    "date": "2024-11-21T10:01:00"
                }
            ]
        }

        messages = parser.extract_messages(data)

        assert len(messages) == 1
        assert messages[0]["external_id"] == "1"

    def test_extract_messages_filters_empty_content(self):
        """Test that messages with empty content are filtered."""
        parser = TelegramParser()

        data = {
            "messages": [
                {
                    "id": 1,
                    "type": "message",
                    "text": "",
                    "from": "John",
                    "date": "2024-11-21T10:00:00"
                }
            ]
        }

        messages = parser.extract_messages(data)
        assert len(messages) == 0

    def test_extract_messages_complete_message(self):
        """Test extracting a complete message."""
        parser = TelegramParser()

        data = {
            "messages": [
                {
                    "id": 123,
                    "type": "message",
                    "text": "Test message",
                    "from": "John Doe",
                    "from_id": "user456",
                    "date": "2024-11-21T10:30:45"
                }
            ]
        }

        messages = parser.extract_messages(data)

        assert len(messages) == 1
        msg = messages[0]
        assert msg["external_id"] == "123"
        assert msg["content"] == "Test message"
        assert msg["author_name"] == "John Doe"
        assert msg["author_id"] == "user456"
        assert msg["author_type"] == "user"
        assert msg["is_public"] is True
        assert isinstance(msg["published_at"], datetime)

    def test_extract_messages_empty_list(self):
        """Test extracting from empty messages list."""
        parser = TelegramParser()

        data = {"messages": []}
        messages = parser.extract_messages(data)
        assert len(messages) == 0

    def test_extract_messages_missing_field(self):
        """Test extracting when messages field is missing."""
        parser = TelegramParser()

        data = {}
        messages = parser.extract_messages(data)
        assert len(messages) == 0

    def test_extract_messages_invalid_type(self):
        """Test extracting when messages is not a list."""
        parser = TelegramParser()

        data = {"messages": "not-a-list"}
        messages = parser.extract_messages(data)
        assert len(messages) == 0


class TestTelegramParserIntegration:
    """Integration tests for TelegramParser file operations."""

    def test_parse_file_not_found(self):
        """Test parsing non-existent file raises error."""
        parser = TelegramParser()

        with pytest.raises(FileNotFoundError):
            parser.parse_file("/nonexistent/file.json")

    def test_parse_file_requires_path(self):
        """Test that parse_file requires a path."""
        parser = TelegramParser()

        # Should work with Path object
        with pytest.raises(FileNotFoundError):
            parser.parse_file(Path("/nonexistent/file.json"))
