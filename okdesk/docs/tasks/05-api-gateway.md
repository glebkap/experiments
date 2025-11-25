# API Gateway Service - Декомпозиция задач

**Task ID:** 05-api-gateway
**Приоритет:** Средний
**Дата:** 25.11.2025
**Технологии:** Python 3.12, FastAPI, httpx
**Зависит от:** Parser Service, Analyzer Service

---

## Цель

Реализовать простой API Gateway как единую точку входа для всех клиентов (GUI, CLI). Gateway выполняет роль **reverse proxy** и не содержит бизнес-логики.

**Принципы:**

- Максимальная простота
- Только роутинг к сервисам
- Никакой бизнес-логики
- Агрегация health checks

---

## Архитектура

```
┌──────────┐
│  Client  │ (GUI / CLI / curl)
└────┬─────┘
     │ HTTP :8000
     ▼
┌─────────────────┐
│  API Gateway    │ FastAPI
│  (port 8000)    │
└────┬────────────┘
     │
     ├─────────► Parser Service    (port 8001)
     └─────────► Analyzer Service  (port 8002)
```

---

## Структура проекта

```
services/api-gateway/
├── src/
│   ├── config.py              # Конфигурация (URLs сервисов)
│   ├── proxy.py               # Reverse proxy логика
│   ├── health.py              # Health check aggregation
│   ├── main.py                # FastAPI app
│   └── __init__.py
├── tests/
│   ├── test_proxy.py
│   └── test_health.py
├── Dockerfile
├── pyproject.toml
├── .dockerignore
└── README.md
```

---

## Реализация

### 1. Configuration (`src/config.py`)

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Service URLs
    parser_url: str = "http://parser:8001"
    analyzer_url: str = "http://analyzer:8002"

    # API Gateway settings
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS
    cors_origins: list[str] = ["http://localhost:3000"]  # GUI

    # Timeouts
    request_timeout: int = 30  # seconds

    class Config:
        env_file = ".env"

settings = Settings()
```

### 2. Reverse Proxy (`src/proxy.py`)

```python
import httpx
from fastapi import Request, Response, HTTPException
from fastapi.responses import StreamingResponse
import logging

logger = logging.getLogger(__name__)

async def proxy_request(
    request: Request,
    target_url: str,
    timeout: int = 30
) -> Response:
    """
    Проксировать запрос к целевому сервису.

    Args:
        request: Incoming FastAPI request
        target_url: Target service base URL
        timeout: Request timeout in seconds

    Returns:
        Response from target service
    """
    # Build full target URL
    path = request.url.path
    query = str(request.url.query)
    full_url = f"{target_url}{path}"
    if query:
        full_url = f"{full_url}?{query}"

    logger.info(f"Proxying {request.method} {path} -> {full_url}")

    # Prepare headers (remove host header)
    headers = dict(request.headers)
    headers.pop("host", None)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Forward request
            response = await client.request(
                method=request.method,
                url=full_url,
                headers=headers,
                content=await request.body(),
                follow_redirects=False
            )

            # Return response
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.headers.get("content-type")
            )

    except httpx.TimeoutException:
        logger.error(f"Timeout proxying to {target_url}")
        raise HTTPException(status_code=504, detail="Gateway Timeout")

    except httpx.RequestError as e:
        logger.error(f"Error proxying to {target_url}: {e}")
        raise HTTPException(status_code=503, detail="Service Unavailable")
```

### 3. Health Check Aggregation (`src/health.py`)

```python
import httpx
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

async def check_service_health(url: str, timeout: int = 5) -> Dict[str, Any]:
    """
    Проверить health endpoint сервиса.

    Returns:
        {
            "status": "ok" | "error",
            "latency_ms": float,
            "details": {...}
        }
    """
    import time

    health_url = f"{url}/health"
    start = time.time()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(health_url)
            latency = (time.time() - start) * 1000

            if response.status_code == 200:
                return {
                    "status": "ok",
                    "latency_ms": round(latency, 2),
                    "details": response.json() if response.content else {}
                }
            else:
                return {
                    "status": "error",
                    "latency_ms": round(latency, 2),
                    "details": {"status_code": response.status_code}
                }

    except Exception as e:
        latency = (time.time() - start) * 1000
        logger.error(f"Health check failed for {url}: {e}")
        return {
            "status": "error",
            "latency_ms": round(latency, 2),
            "details": {"error": str(e)}
        }

async def aggregate_health_checks(service_urls: Dict[str, str]) -> Dict[str, Any]:
    """
    Агрегировать health checks всех сервисов.

    Returns:
        {
            "status": "ok" | "degraded" | "error",
            "services": {
                "parser": {...},
                "analyzer": {...}
            }
        }
    """
    import asyncio

    # Check all services concurrently
    tasks = {
        name: check_service_health(url)
        for name, url in service_urls.items()
    }

    results = {}
    for name, task in tasks.items():
        results[name] = await task

    # Determine overall status
    all_ok = all(r["status"] == "ok" for r in results.values())
    any_ok = any(r["status"] == "ok" for r in results.values())

    if all_ok:
        overall_status = "ok"
    elif any_ok:
        overall_status = "degraded"
    else:
        overall_status = "error"

    return {
        "status": overall_status,
        "services": results
    }
```

### 4. Main Application (`src/main.py`)

```python
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from .config import settings
from .proxy import proxy_request
from .health import aggregate_health_checks

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Support System API Gateway",
    version="1.0.0",
    description="Reverse proxy to Parser and Analyzer services"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Health Endpoints ====================
@app.get("/health")
async def health_check():
    """Gateway health check"""
    return {"status": "ok", "service": "api-gateway"}

@app.get("/api/v1/health")
async def aggregated_health():
    """Aggregated health check of all services"""
    service_urls = {
        "parser": settings.parser_url,
        "analyzer": settings.analyzer_url
    }
    return await aggregate_health_checks(service_urls)

@app.get("/api/v1/health/parser")
async def parser_health():
    """Parser service health"""
    from .health import check_service_health
    return await check_service_health(settings.parser_url)

@app.get("/api/v1/health/analyzer")
async def analyzer_health():
    """Analyzer service health"""
    from .health import check_service_health
    return await check_service_health(settings.analyzer_url)

# ==================== Proxy Routes ====================
@app.api_route("/api/v1/import/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_to_parser(request: Request):
    """Proxy all /import/* requests to Parser Service"""
    return await proxy_request(
        request,
        settings.parser_url,
        timeout=settings.request_timeout
    )

@app.api_route("/api/v1/analyzer/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_to_analyzer(request: Request):
    """Proxy all /analyzer/* requests to Analyzer Service"""
    return await proxy_request(
        request,
        settings.analyzer_url,
        timeout=settings.request_timeout
    )

# ==================== Root ====================
@app.get("/")
async def root():
    """API Gateway info"""
    return {
        "service": "api-gateway",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "aggregated_health": "/api/v1/health",
            "parser": "/api/v1/import/*",
            "analyzer": "/api/v1/analyzer/*"
        }
    }

# ==================== Error Handlers ====================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"}
    )
```

### 5. Dependencies (`pyproject.toml`)

```toml
[project]
name = "api-gateway-service"
version = "1.0.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.23.0",
    "httpx>=0.24.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "httpx>=0.24.0",
]
```

### 6. Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/

# Install dependencies
RUN uv pip install --system -e .

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 7. Environment Variables (`.env`)

```env
# Service URLs
PARSER_URL=http://parser:8001
ANALYZER_URL=http://analyzer:8002

# API Gateway
HOST=0.0.0.0
PORT=8000

# CORS
CORS_ORIGINS=["http://localhost:3000","http://localhost:8000"]

# Timeouts
REQUEST_TIMEOUT=30
```

---

## Docker Compose Integration

Добавить в главный `docker-compose.yml`:

```yaml
api-gateway:
  build: ./services/api-gateway
  container_name: support-api-gateway
  depends_on:
    - parser
    - analyzer
  environment:
    PARSER_URL: http://parser:8001
    ANALYZER_URL: http://analyzer:8002
  ports:
    - "8000:8000"
  networks:
    - support-network
```

---

## Тестирование

### Unit Tests (`tests/test_proxy.py`)

```python
import pytest
from httpx import AsyncClient
from src.main import app

@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

@pytest.mark.asyncio
async def test_root():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/")
        assert response.status_code == 200
        assert "service" in response.json()
```

---

## Критерии готовности

- [ ] API Gateway запускается
- [ ] Health checks работают
- [ ] Proxy к Parser работает
- [ ] Proxy к Analyzer работает
- [ ] CORS настроен
- [ ] Timeouts работают корректно
- [ ] Тесты проходят
- [ ] Dockerfile готов

---

## Оценка времени

- Реализация proxy + health: **3-4 часа**
- Тестирование: **1-2 часа**
- Docker integration: **1 час**

**Итого: 1 день**

---

## Примечания

1. **Простота** - это главное преимущество. Gateway не должен усложняться.
2. **No business logic** - только routing и health checks.
3. **Error handling** - 503 если сервис недоступен, 504 при timeout.
4. **Logging** - логировать все proxy запросы для отладки.
5. **Future improvements** - rate limiting, authentication (опционально).
