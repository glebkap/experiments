.PHONY: help install test test-unit test-integration lint format clean run docker-build docker-run

help:
	@echo "Analyzer Service - Makefile Commands"
	@echo ""
	@echo "Available targets:"
	@echo "  install           - Install dependencies with uv"
	@echo "  test              - Run all tests with coverage"
	@echo "  test-unit         - Run only unit tests"
	@echo "  test-integration  - Run only integration tests"
	@echo "  lint              - Run linters (ruff)"
	@echo "  format            - Format code with ruff"
	@echo "  clean             - Clean cache and temporary files"
	@echo "  run               - Run the service locally"
	@echo "  docker-build      - Build Docker image"
	@echo "  docker-run        - Run service in Docker"

install:
	uv sync

test:
	uv run pytest -v --cov=src --cov-report=term-missing --cov-report=html

test-unit:
	uv run pytest tests/unit -v -m unit

test-integration:
	uv run pytest tests/integration -v -m integration

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

run:
	uv run python -m uvicorn src.main:app --host 0.0.0.0 --port 8002 --reload

docker-build:
	docker build -t analyzer-service:latest .

docker-run:
	docker run -p 8002:8002 --env-file .env analyzer-service:latest
