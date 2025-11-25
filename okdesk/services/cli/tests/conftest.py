"""Pytest configuration and fixtures."""

from unittest.mock import Mock

import pytest

from support_cli.client.api_client import APIClient
from support_cli.config import Settings


@pytest.fixture
def mock_settings() -> Settings:
    """Create mock settings for testing.

    Returns:
        Settings: Mock settings instance
    """
    return Settings(
        api_gateway_url="http://test-api:8000",  # type: ignore
        request_timeout=10,
        max_retries=1,
    )


@pytest.fixture
def mock_api_client(mock_settings: Settings) -> Mock:
    """Create mock API client for testing.

    Args:
        mock_settings: Mock settings fixture

    Returns:
        Mock: Mock API client
    """
    client = Mock(spec=APIClient)
    client.settings = mock_settings
    return client
