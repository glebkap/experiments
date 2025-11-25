# Multi-stage build for API Gateway Service

FROM python:3.12-slim as builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies
RUN uv sync --frozen --no-dev || uv pip install --system fastapi uvicorn httpx pydantic pydantic-settings

# Production stage
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy uv and dependencies from builder
COPY --from=builder /app/.venv /app/.venv 2>/dev/null || true
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# If .venv doesn't exist, install directly to system
RUN if [ ! -d "/app/.venv" ]; then \
        uv pip install --system fastapi uvicorn httpx pydantic pydantic-settings; \
    fi

# Copy application code
COPY src /app/src

# Set Python path
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
