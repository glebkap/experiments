"""Unit tests for TextPreprocessor."""

import pytest

from domain.services.text_preprocessor import TextPreprocessor


class TestTextPreprocessor:
    """Test suite for TextPreprocessor."""

    @pytest.fixture
    def preprocessor(self):
        """Create TextPreprocessor instance."""
        return TextPreprocessor()

    def test_clean_html(self, preprocessor):
        """Test HTML cleaning."""
        html_text = "<p>Hello <b>world</b>!</p>"
        result = preprocessor.preprocess(html_text)

        assert "<p>" not in result
        assert "<b>" not in result
        assert "hello" in result
        assert "world" in result

    def test_normalize_url(self, preprocessor):
        """Test URL normalization."""
        text = "Visit https://example.com for more info"
        result = preprocessor.preprocess(text)

        # URL should be normalized to [url] (lowercase after processing)
        assert "[url]" in result
        assert "https://example.com" not in result

    def test_normalize_email(self, preprocessor):
        """Test email normalization."""
        text = "Contact us at test@example.com"
        result = preprocessor.preprocess(text)

        # Email should be normalized to [email] (lowercase after processing)
        assert "[email]" in result
        assert "test@example.com" not in result

    def test_normalize_phone(self, preprocessor):
        """Test phone normalization."""
        text = "Call +7 (123) 456-78-90"
        result = preprocessor.preprocess(text)

        # Phone should be normalized to [phone] (lowercase after processing)
        assert "[phone]" in result

    def test_lowercase(self, preprocessor):
        """Test lowercase conversion."""
        text = "Hello WORLD"
        result = preprocessor.preprocess(text)

        assert result == result.lower()

    def test_empty_text(self, preprocessor):
        """Test empty text handling."""
        assert preprocessor.preprocess("") == ""
        assert preprocessor.preprocess(None) == ""

    def test_preprocess_issue(self, preprocessor):
        """Test issue preprocessing with title and description."""
        title = "Bug Report"
        description = "<p>Description with <b>HTML</b></p>"

        result = preprocessor.preprocess_issue(title, description)

        assert "bug" in result.lower()
        assert "report" in result.lower()
        assert "description" in result.lower()
        assert "<p>" not in result
