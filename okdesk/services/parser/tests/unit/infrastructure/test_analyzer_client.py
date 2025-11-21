"""Unit tests for AnalyzerClient."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
import httpx

from src.infrastructure.http import AnalyzerClient


class TestAnalyzerClient:
    """Tests for AnalyzerClient."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test client initialization."""
        client = AnalyzerClient("http://analyzer:8002", timeout=10.0, max_retries=5)

        assert client._base_url == "http://analyzer:8002"
        assert client._timeout == 10.0
        assert client._max_retries == 5

    @pytest.mark.asyncio
    async def test_base_url_strips_trailing_slash(self):
        """Test that trailing slash is removed from base URL."""
        client = AnalyzerClient("http://analyzer:8002/", timeout=30.0)

        assert client._base_url == "http://analyzer:8002"

    @pytest.mark.asyncio
    async def test_analyze_batch_success(self):
        """Test successful batch analysis request."""
        client = AnalyzerClient("http://analyzer:8002")

        message_ids = [uuid4(), uuid4(), uuid4()]
        expected_response = {"total": 3, "processed": 3, "status": "completed"}

        # Mock HTTP client
        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json = MagicMock(return_value=expected_response)
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            result = await client.analyze_batch(message_ids, batch_size=10)

            assert result == expected_response
            mock_post.assert_called_once()

            # Check request payload
            call_kwargs = mock_post.call_args.kwargs
            assert "json" in call_kwargs
            payload = call_kwargs["json"]
            assert len(payload["message_ids"]) == 3
            assert payload["batch_size"] == 10

    @pytest.mark.asyncio
    async def test_analyze_batch_converts_uuids_to_strings(self):
        """Test that UUIDs are converted to strings in request."""
        client = AnalyzerClient("http://analyzer:8002")

        message_ids = [uuid4(), uuid4()]

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json = MagicMock(return_value= {})
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            await client.analyze_batch(message_ids, batch_size=5)

            payload = mock_post.call_args.kwargs["json"]
            # All IDs should be strings
            for msg_id in payload["message_ids"]:
                assert isinstance(msg_id, str)

    @pytest.mark.asyncio
    async def test_analyze_batch_retry_on_server_error(self):
        """Test retry on server error (5xx)."""
        client = AnalyzerClient("http://analyzer:8002", max_retries=3)

        message_ids = [uuid4()]

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            # First two calls fail, third succeeds
            mock_error = httpx.HTTPStatusError(
                "Server error",
                request=AsyncMock(),
                response=AsyncMock(status_code=500)
            )

            mock_success = MagicMock()
            mock_success.status_code = 200
            mock_success.json = MagicMock(return_value={"status": "ok"})
            mock_success.raise_for_status = MagicMock()

            mock_post.side_effect = [
                mock_error,
                mock_error,
                mock_success
            ]

            result = await client.analyze_batch(message_ids)

            assert result == {"status": "ok"}
            assert mock_post.call_count == 3

    @pytest.mark.asyncio
    async def test_analyze_batch_no_retry_on_client_error(self):
        """Test no retry on client error (4xx)."""
        client = AnalyzerClient("http://analyzer:8002", max_retries=3)

        message_ids = [uuid4()]

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_response = AsyncMock(status_code=400)
            mock_error = httpx.HTTPStatusError(
                "Bad request",
                request=AsyncMock(),
                response=mock_response
            )
            mock_post.side_effect = mock_error

            with pytest.raises(httpx.HTTPStatusError):
                await client.analyze_batch(message_ids)

            # Should only call once (no retries for 4xx)
            assert mock_post.call_count == 1

    @pytest.mark.asyncio
    async def test_analyze_batch_fails_after_max_retries(self):
        """Test that it fails after max retries exhausted."""
        client = AnalyzerClient("http://analyzer:8002", max_retries=2)

        message_ids = [uuid4()]

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_error = httpx.HTTPError("Connection error")
            mock_post.side_effect = mock_error

            with pytest.raises(httpx.HTTPError):
                await client.analyze_batch(message_ids)

            # Should try max_retries times
            assert mock_post.call_count == 2

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        client = AnalyzerClient("http://analyzer:8002")

        with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            result = await client.health_check()

            assert result is True
            mock_get.assert_called_once_with("http://analyzer:8002/health", timeout=5.0)

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test health check failure."""
        client = AnalyzerClient("http://analyzer:8002")

        with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 503
            mock_get.return_value = mock_response

            result = await client.health_check()

            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_exception(self):
        """Test health check with exception."""
        client = AnalyzerClient("http://analyzer:8002")

        with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection refused")

            result = await client.health_check()

            assert result is False

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing the client."""
        client = AnalyzerClient("http://analyzer:8002")

        with patch.object(client._client, "aclose", new_callable=AsyncMock) as mock_close:
            await client.close()

            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test using client as async context manager."""
        with patch("src.infrastructure.http.analyzer_client.httpx.AsyncClient") as mock_client_cls:
            # Mock the AsyncClient instance
            mock_client_instance = AsyncMock()
            mock_client_cls.return_value = mock_client_instance

            async with AnalyzerClient("http://analyzer:8002") as client:
                assert client is not None

            # Close should be called automatically
            mock_client_instance.aclose.assert_called_once()
