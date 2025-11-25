"""Reverse proxy functionality for API Gateway."""

import logging
from typing import Any

import httpx
from fastapi import HTTPException, Request, Response

logger = logging.getLogger(__name__)


async def proxy_request(
    request: Request,
    target_url: str,
    timeout: int = 30,
) -> Response:
    """
    Proxy HTTP request to target service.

    Args:
        request: Incoming FastAPI request
        target_url: Target service base URL (e.g., http://parser:8001)
        timeout: Request timeout in seconds

    Returns:
        Response from target service

    Raises:
        HTTPException: On timeout (504) or connection error (503)
    """
    # Build full target URL with path and query
    path = request.url.path
    query = str(request.url.query)
    full_url = f"{target_url}{path}"
    if query:
        full_url = f"{full_url}?{query}"

    logger.info(f"Proxying {request.method} {path} -> {full_url}")

    # Prepare headers (remove host header to avoid conflicts)
    headers = dict(request.headers)
    headers.pop("host", None)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Forward request to target service
            response = await client.request(
                method=request.method,
                url=full_url,
                headers=headers,
                content=await request.body(),
                follow_redirects=False,
            )

            # Return response with same status code and headers
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.headers.get("content-type"),
            )

    except httpx.TimeoutException as e:
        logger.error(f"Timeout proxying {request.method} {path} to {target_url}: {e}")
        raise HTTPException(
            status_code=504,
            detail=f"Gateway Timeout: {target_url} did not respond in time",
        )

    except httpx.RequestError as e:
        logger.error(f"Error proxying {request.method} {path} to {target_url}: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Service Unavailable: Cannot connect to {target_url}",
        )
