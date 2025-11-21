"""HTTP client for Analyzer Service."""

import logging
from typing import Any
from uuid import UUID

import httpx

logger = logging.getLogger(__name__)


class AnalyzerClient:
    """Client for communicating with Analyzer Service."""

    def __init__(self, base_url: str, timeout: float = 30.0, max_retries: int = 3) -> None:
        """
        Initialize Analyzer client.

        Args:
            base_url: Base URL of Analyzer Service
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.aclose()

    async def analyze_batch(
        self, message_ids: list[UUID], batch_size: int = 10
    ) -> dict[str, Any]:
        """
        Request batch analysis of messages.

        Args:
            message_ids: List of message UUIDs to analyze
            batch_size: Number of messages per batch

        Returns:
            Response dictionary from Analyzer Service

        Raises:
            httpx.HTTPError: If request fails after retries
        """
        url = f"{self._base_url}/api/v1/analyze/batch"
        payload = {
            "message_ids": [str(mid) for mid in message_ids],
            "batch_size": batch_size,
        }

        logger.info(
            f"Requesting analysis of {len(message_ids)} messages " f"(batch_size={batch_size})"
        )

        # Try with retries
        last_error = None
        for attempt in range(self._max_retries):
            try:
                response = await self._client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()

                logger.info(f"Analysis request successful: {result}")
                return result

            except httpx.HTTPError as e:
                last_error = e
                logger.warning(
                    f"Analysis request failed (attempt {attempt + 1}/{self._max_retries}): {e}"
                )

                # Don't retry on client errors (4xx)
                if isinstance(e, httpx.HTTPStatusError) and 400 <= e.response.status_code < 500:
                    logger.error(f"Client error, not retrying: {e}")
                    raise

                # Wait before retry (exponential backoff)
                if attempt < self._max_retries - 1:
                    import asyncio

                    wait_time = 2**attempt  # 1s, 2s, 4s
                    logger.info(f"Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)

        # All retries failed
        logger.error(f"All {self._max_retries} attempts failed")
        if last_error:
            raise last_error
        raise RuntimeError("Analysis request failed without specific error")

    async def health_check(self) -> bool:
        """
        Check if Analyzer Service is healthy.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            url = f"{self._base_url}/health"
            response = await self._client.get(url, timeout=5.0)
            return response.status_code == 200
        except httpx.HTTPError as e:
            logger.warning(f"Health check failed: {e}")
            return False

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
