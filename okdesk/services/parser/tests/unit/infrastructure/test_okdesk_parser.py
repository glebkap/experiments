"""Unit tests for OKDesk parser."""

from datetime import datetime
from src.infrastructure.parsers import OKDeskParser


class TestOKDeskParser:
    """Tests for OKDeskParser."""

    def test_clean_html(self):
        """Test HTML cleaning."""
        parser = OKDeskParser()

        # Test with HTML
        html = "<p>Hello <strong>world</strong>!</p>"
        result = parser._clean_html(html)
        assert "Hello world" in result  # BeautifulSoup may add spaces

        # Test with plain text
        text = "Plain text"
        result = parser._clean_html(text)
        assert result == "Plain text"

        # Test with None
        result = parser._clean_html(None)
        assert result is None

    def test_map_status(self):
        """Test status mapping."""
        parser = OKDeskParser()

        assert parser._map_status("opened") == "opened"
        assert parser._map_status("open") == "opened"
        assert parser._map_status("wait") == "wait"
        assert parser._map_status("completed") == "completed"
        assert parser._map_status("closed") == "closed"
        assert parser._map_status("unknown") is None
        assert parser._map_status(None) is None

    def test_map_priority(self):
        """Test priority mapping."""
        parser = OKDeskParser()

        assert parser._map_priority(1) == 1
        assert parser._map_priority(4) == 4
        assert parser._map_priority(0) is None  # Out of range
        assert parser._map_priority(5) is None  # Out of range
        assert parser._map_priority("invalid") is None
        assert parser._map_priority(None) is None

    def test_map_author_type(self):
        """Test author type mapping."""
        parser = OKDeskParser()

        assert parser._map_author_type("employee") == "employee"
        assert parser._map_author_type("contact") == "contact"
        assert parser._map_author_type("client") == "contact"
        assert parser._map_author_type("user") == "user"
        assert parser._map_author_type("unknown") is None
        assert parser._map_author_type(None) is None

    def test_parse_datetime(self):
        """Test datetime parsing."""
        parser = OKDeskParser()

        # Test ISO format
        dt = parser._parse_datetime("2024-11-21T10:30:00Z")
        assert isinstance(dt, datetime)
        assert dt.year == 2024
        assert dt.month == 11

        # Test invalid format
        result = parser._parse_datetime("invalid")
        assert result is None

        # Test None
        result = parser._parse_datetime(None)
        assert result is None

    def test_extract_issue(self):
        """Test extracting issue data."""
        parser = OKDeskParser()

        data = {
            "id": 12345,
            "title": "<p>Test Issue</p>",
            "description": "<div>Description</div>",
            "status": "opened",
            "priority": 2,
            "created_at": "2024-11-21T10:00:00Z",
        }

        result = parser.extract_issue(data)

        assert result["external_id"] == "12345"
        assert result["title"] == "Test Issue"
        assert result["description"] == "Description"
        assert result["status"] == "opened"
        assert result["priority"] == 2

    def test_extract_comments(self):
        """Test extracting comments."""
        parser = OKDeskParser()

        data = {
            "comments": [
                {
                    "id": 1,
                    "content": "<p>First comment</p>",
                    "author": {"id": 100, "name": "John", "type": "employee"},
                    "public": True,
                    "published_at": "2024-11-21T10:00:00Z",
                },
                {
                    "id": 2,
                    "content": "Second comment",
                    "author": {"id": 200, "name": "Jane", "type": "contact"},
                    "public": False,
                },
            ]
        }

        result = parser.extract_comments(data)

        assert len(result) == 2
        assert result[0]["external_id"] == "1"
        assert result[0]["content"] == "First comment"
        assert result[0]["author_name"] == "John"
        assert result[0]["author_type"] == "employee"
        assert result[0]["is_public"] is True
