# Multi-stage build for CLI service
FROM python:3.12-slim AS builder

# Install uv
RUN pip install uv

WORKDIR /build

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/

# Install dependencies
RUN uv pip install --system -e .

# Runtime stage
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/support-cli /usr/local/bin/support-cli

# Copy source code
COPY src/ ./src/
COPY .env.example ./.env

# Create volume mount point for data
VOLUME ["/app/data"]

# Set environment variable
ENV PYTHONUNBUFFERED=1

# Entry point
ENTRYPOINT ["support-cli"]
CMD ["--help"]
