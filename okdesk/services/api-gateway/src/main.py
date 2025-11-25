"""Main FastAPI application for API Gateway."""

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .health import aggregate_health_checks, check_service_health
from .proxy import proxy_request

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Support System API Gateway",
    version="1.0.0",
    description="Reverse proxy to Parser and Analyzer services",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Health Check Endpoints ====================


@app.get("/health")
async def gateway_health():
    """
    API Gateway health check.

    Returns:
        Status of the gateway itself (always ok if running).
    """
    return {
        "status": "ok",
        "service": "api-gateway",
        "version": "1.0.0",
    }


@app.get("/api/v1/health")
async def aggregated_health():
    """
    Aggregated health check of all backend services.

    Returns:
        Overall status and individual service health checks.

    Response format:
        {
            "status": "ok" | "degraded" | "error",
            "services": {
                "parser": {"status": "ok", "latency_ms": 12.5, ...},
                "analyzer": {"status": "ok", "latency_ms": 15.3, ...}
            }
        }
    """
    service_urls = {
        "parser": settings.parser_url,
        "analyzer": settings.analyzer_url,
    }
    return await aggregate_health_checks(
        service_urls,
        timeout=settings.health_check_timeout,
    )


@app.get("/api/v1/health/parser")
async def parser_health():
    """
    Parser service health check.

    Returns:
        Health status of Parser service.
    """
    return await check_service_health(
        settings.parser_url,
        timeout=settings.health_check_timeout,
    )


@app.get("/api/v1/health/analyzer")
async def analyzer_health():
    """
    Analyzer service health check.

    Returns:
        Health status of Analyzer service.
    """
    return await check_service_health(
        settings.analyzer_url,
        timeout=settings.health_check_timeout,
    )


# ==================== Proxy Routes ====================


@app.api_route(
    "/api/v1/import/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
)
async def proxy_to_parser(request: Request):
    """
    Proxy all /import/* requests to Parser Service.

    This route handles:
    - POST /api/v1/import/okdesk - Upload OKDesk data
    - POST /api/v1/import/telegram - Upload Telegram data
    - GET /api/v1/import/{id} - Get import status
    - GET /api/v1/import - List imports

    All requests are forwarded to Parser Service at port 8001.
    """
    return await proxy_request(
        request,
        settings.parser_url,
        timeout=settings.request_timeout,
    )


@app.api_route(
    "/api/v1/analyzer/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
)
async def proxy_to_analyzer(request: Request):
    """
    Proxy all /analyzer/* requests to Analyzer Service.

    This route handles:
    - POST /api/v1/analyzer/pipeline/process - Run processing pipeline
    - GET /api/v1/analyzer/pipeline/status - Pipeline status
    - POST /api/v1/analyzer/clustering/run - Run clustering
    - GET /api/v1/analyzer/clustering/info - Cluster information
    - POST /api/v1/analyzer/search/similar - Semantic search
    - GET /api/v1/analyzer/issues - List issues
    - GET /api/v1/analyzer/issues/{id} - Issue details
    - GET /api/v1/analyzer/clusters/{id}/issues - Issues in cluster
    - GET /api/v1/analyzer/search/fulltext - Full-text search
    - GET /api/v1/analyzer/stats/* - Statistics endpoints
    - POST /api/v1/analyzer/export - Export data

    All requests are forwarded to Analyzer Service at port 8002.
    """
    return await proxy_request(
        request,
        settings.analyzer_url,
        timeout=settings.request_timeout,
    )


# ==================== Root Endpoint ====================


@app.get("/")
async def root():
    """
    API Gateway information and available endpoints.

    Returns:
        Gateway metadata and route documentation.
    """
    return {
        "service": "api-gateway",
        "version": "1.0.0",
        "description": "Support System API Gateway - Reverse proxy to microservices",
        "endpoints": {
            "health": {
                "gateway": "/health",
                "all_services": "/api/v1/health",
                "parser": "/api/v1/health/parser",
                "analyzer": "/api/v1/health/analyzer",
            },
            "proxied_routes": {
                "parser": "/api/v1/import/*",
                "analyzer": "/api/v1/analyzer/*",
            },
            "documentation": {
                "swagger": "/docs",
                "redoc": "/redoc",
            },
        },
        "configuration": {
            "parser_url": settings.parser_url,
            "analyzer_url": settings.analyzer_url,
            "request_timeout": f"{settings.request_timeout}s",
        },
    }


# ==================== Error Handlers ====================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors.

    Logs the error and returns a 500 Internal Server Error response.
    """
    logger.error(
        f"Unhandled exception in {request.method} {request.url.path}: {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal Server Error",
            "path": str(request.url.path),
        },
    )


# ==================== Startup/Shutdown Events ====================


@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    logger.info("API Gateway starting up...")
    logger.info(f"Parser Service URL: {settings.parser_url}")
    logger.info(f"Analyzer Service URL: {settings.analyzer_url}")
    logger.info(f"CORS Origins: {settings.cors_origins}")
    logger.info(f"Request Timeout: {settings.request_timeout}s")
    logger.info("API Gateway ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown information."""
    logger.info("API Gateway shutting down...")
