"""Tests for health check functionality."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.health import check_service_health, aggregate_health_checks


@pytest.mark.asyncio
async def test_check_service_health_ok():
    """Test successful health check."""
    # Mock httpx client response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b'{"status": "ok", "service": "parser"}'
    mock_response.json.return_value = {"status": "ok", "service": "parser"}

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        result = await check_service_health("http://parser:8001", timeout=5)

    assert result["status"] == "ok"
    assert result["latency_ms"] >= 0
    assert "details" in result
    assert result["details"]["service"] == "parser"


@pytest.mark.asyncio
async def test_check_service_health_error_status():
    """Test health check with non-200 status code."""
    # Mock httpx client response with error status
    mock_response = MagicMock()
    mock_response.status_code = 500

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        result = await check_service_health("http://parser:8001", timeout=5)

    assert result["status"] == "error"
    assert result["latency_ms"] >= 0
    assert result["details"]["status_code"] == 500


@pytest.mark.asyncio
async def test_check_service_health_timeout():
    """Test health check timeout."""
    from httpx import TimeoutException

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(side_effect=TimeoutException("Timeout"))
        mock_client_class.return_value = mock_client

        result = await check_service_health("http://parser:8001", timeout=5)

    assert result["status"] == "error"
    assert result["latency_ms"] >= 0
    assert result["details"]["error"] == "timeout"


@pytest.mark.asyncio
async def test_check_service_health_connection_error():
    """Test health check connection error."""
    from httpx import RequestError

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(side_effect=RequestError("Connection refused"))
        mock_client_class.return_value = mock_client

        result = await check_service_health("http://parser:8001", timeout=5)

    assert result["status"] == "error"
    assert result["latency_ms"] >= 0
    assert "error" in result["details"]


@pytest.mark.asyncio
async def test_aggregate_health_all_ok():
    """Test aggregate health when all services are ok."""
    service_urls = {
        "parser": "http://parser:8001",
        "analyzer": "http://analyzer:8002",
    }

    # Mock httpx client responses
    mock_response_ok = MagicMock()
    mock_response_ok.status_code = 200
    mock_response_ok.content = b'{"status": "ok"}'
    mock_response_ok.json.return_value = {"status": "ok"}

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(return_value=mock_response_ok)
        mock_client_class.return_value = mock_client

        result = await aggregate_health_checks(service_urls, timeout=5)

    assert result["status"] == "ok"
    assert "parser" in result["services"]
    assert "analyzer" in result["services"]
    assert result["services"]["parser"]["status"] == "ok"
    assert result["services"]["analyzer"]["status"] == "ok"


@pytest.mark.asyncio
async def test_aggregate_health_degraded():
    """Test aggregate health when one service is down."""
    service_urls = {
        "parser": "http://parser:8001",
        "analyzer": "http://analyzer:8002",
    }

    # Mock responses: parser ok, analyzer error
    mock_response_ok = MagicMock()
    mock_response_ok.status_code = 200
    mock_response_ok.content = b'{"status": "ok"}'
    mock_response_ok.json.return_value = {"status": "ok"}

    from httpx import RequestError

    call_count = 0

    async def mock_get(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return mock_response_ok
        else:
            raise RequestError("Connection refused")

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = mock_get
        mock_client_class.return_value = mock_client

        result = await aggregate_health_checks(service_urls, timeout=5)

    assert result["status"] == "degraded"
    assert "parser" in result["services"]
    assert "analyzer" in result["services"]


@pytest.mark.asyncio
async def test_aggregate_health_all_error():
    """Test aggregate health when all services are down."""
    service_urls = {
        "parser": "http://parser:8001",
        "analyzer": "http://analyzer:8002",
    }

    from httpx import RequestError

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(side_effect=RequestError("Connection refused"))
        mock_client_class.return_value = mock_client

        result = await aggregate_health_checks(service_urls, timeout=5)

    assert result["status"] == "error"
    assert result["services"]["parser"]["status"] == "error"
    assert result["services"]["analyzer"]["status"] == "error"


@pytest.mark.asyncio
async def test_aggregated_health_endpoint():
    """Test /api/v1/health endpoint."""
    from fastapi.testclient import TestClient
    from src.main import app

    # Mock check_service_health to return ok status
    with patch("src.main.check_service_health") as mock_check:
        mock_check.return_value = {
            "status": "ok",
            "latency_ms": 10.5,
            "details": {},
        }

        client = TestClient(app)
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "services" in data


@pytest.mark.asyncio
async def test_parser_health_endpoint():
    """Test /api/v1/health/parser endpoint."""
    from fastapi.testclient import TestClient
    from src.main import app

    # Mock check_service_health
    with patch("src.main.check_service_health") as mock_check:
        mock_check.return_value = {
            "status": "ok",
            "latency_ms": 8.3,
            "details": {"service": "parser"},
        }

        client = TestClient(app)
        response = client.get("/api/v1/health/parser")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_analyzer_health_endpoint():
    """Test /api/v1/health/analyzer endpoint."""
    from fastapi.testclient import TestClient
    from src.main import app

    # Mock check_service_health
    with patch("src.main.check_service_health") as mock_check:
        mock_check.return_value = {
            "status": "ok",
            "latency_ms": 12.7,
            "details": {"service": "analyzer"},
        }

        client = TestClient(app)
        response = client.get("/api/v1/health/analyzer")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
