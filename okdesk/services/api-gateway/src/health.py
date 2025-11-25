"""Health check functionality for API Gateway."""

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


async def check_service_health(
    url: str,
    timeout: int = 5,
) -> dict[str, Any]:
    """
    Check health of a single service.

    Args:
        url: Service base URL (e.g., http://parser:8001)
        timeout: Health check timeout in seconds

    Returns:
        Dictionary with status, latency, and optional details:
        {
            "status": "ok" | "error",
            "latency_ms": float,
            "details": {...} or {"error": str}
        }
    """
    health_url = f"{url}/health"
    start = time.time()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(health_url)
            latency = (time.time() - start) * 1000  # Convert to milliseconds

            if response.status_code == 200:
                # Try to parse JSON response, fallback to empty dict
                try:
                    details = response.json() if response.content else {}
                except Exception:
                    details = {}

                return {
                    "status": "ok",
                    "latency_ms": round(latency, 2),
                    "details": details,
                }
            else:
                return {
                    "status": "error",
                    "latency_ms": round(latency, 2),
                    "details": {"status_code": response.status_code},
                }

    except httpx.TimeoutException:
        latency = (time.time() - start) * 1000
        logger.error(f"Health check timeout for {url}")
        return {
            "status": "error",
            "latency_ms": round(latency, 2),
            "details": {"error": "timeout"},
        }

    except Exception as e:
        latency = (time.time() - start) * 1000
        logger.error(f"Health check failed for {url}: {e}")
        return {
            "status": "error",
            "latency_ms": round(latency, 2),
            "details": {"error": str(e)},
        }


async def aggregate_health_checks(
    service_urls: dict[str, str],
    timeout: int = 5,
) -> dict[str, Any]:
    """
    Aggregate health checks for multiple services.

    Args:
        service_urls: Dictionary mapping service names to URLs
        timeout: Health check timeout per service in seconds

    Returns:
        Dictionary with overall status and per-service results:
        {
            "status": "ok" | "degraded" | "error",
            "services": {
                "parser": {...},
                "analyzer": {...}
            }
        }

        Overall status logic:
        - "ok": All services are healthy
        - "degraded": At least one service is healthy
        - "error": All services are unhealthy
    """
    import asyncio

    # Run all health checks concurrently
    tasks = {
        name: check_service_health(url, timeout)
        for name, url in service_urls.items()
    }

    results = {}
    for name, task in tasks.items():
        results[name] = await task

    # Determine overall status
    statuses = [r["status"] for r in results.values()]
    all_ok = all(status == "ok" for status in statuses)
    any_ok = any(status == "ok" for status in statuses)

    if all_ok:
        overall_status = "ok"
    elif any_ok:
        overall_status = "degraded"
    else:
        overall_status = "error"

    return {
        "status": overall_status,
        "services": results,
    }
