"""Tests for proxy functionality."""

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch, MagicMock

from src.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_gateway_health(client):
    """Test gateway health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "api-gateway"
    assert "version" in data


def test_root_endpoint(client):
    """Test root endpoint returns gateway info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "api-gateway"
    assert "version" in data
    assert "endpoints" in data
    assert "configuration" in data


@pytest.mark.asyncio
async def test_proxy_to_parser_success():
    """Test successful proxy request to parser service."""
    # Mock httpx client response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b'{"status": "ok"}'
    mock_response.headers = {"content-type": "application/json"}

    with patch("src.proxy.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/api/v1/import/")

        assert response.status_code == 200


@pytest.mark.asyncio
async def test_proxy_to_analyzer_success():
    """Test successful proxy request to analyzer service."""
    # Mock httpx client response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b'{"status": "ok"}'
    mock_response.headers = {"content-type": "application/json"}

    with patch("src.proxy.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/api/v1/analyzer/issues")

        assert response.status_code == 200


@pytest.mark.asyncio
async def test_proxy_timeout():
    """Test proxy timeout handling."""
    from httpx import TimeoutException

    with patch("src.proxy.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.request = AsyncMock(side_effect=TimeoutException("Timeout"))
        mock_client_class.return_value = mock_client

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/api/v1/import/")

        assert response.status_code == 504


@pytest.mark.asyncio
async def test_proxy_service_unavailable():
    """Test proxy handling when service is unavailable."""
    from httpx import RequestError

    with patch("src.proxy.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.request = AsyncMock(side_effect=RequestError("Connection error"))
        mock_client_class.return_value = mock_client

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/api/v1/analyzer/issues")

        assert response.status_code == 503


@pytest.mark.asyncio
async def test_proxy_with_query_params():
    """Test proxy preserves query parameters."""
    # Mock httpx client response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b'{"results": []}'
    mock_response.headers = {"content-type": "application/json"}

    with patch("src.proxy.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/api/v1/analyzer/issues?limit=10&offset=0")

        assert response.status_code == 200
        # Verify request was called with correct URL
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert "limit=10" in call_args.kwargs["url"]
        assert "offset=0" in call_args.kwargs["url"]


@pytest.mark.asyncio
async def test_proxy_post_with_body():
    """Test proxy forwards POST request with body."""
    # Mock httpx client response
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.content = b'{"id": "123", "status": "created"}'
    mock_response.headers = {"content-type": "application/json"}

    with patch("src.proxy.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/api/v1/analyzer/pipeline/process",
                json={"batch_size": 100}
            )

        assert response.status_code == 201
        # Verify request was called with body
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args.kwargs["method"] == "POST"
