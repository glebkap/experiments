# ChromaDB Vector Database Makefile

# Configuration
CONTAINER_NAME := support-chromadb
VOLUME_NAME := support-chromadb-data
NETWORK_NAME := support-network
CHROMADB_PORT := 8100
CHROMADB_INTERNAL_PORT := 8000

.PHONY: help build run stop restart logs clean status health network

help:
	@echo "ChromaDB Vector Database Management"
	@echo ""
	@echo "Available targets:"
	@echo "  build    - Build ChromaDB Docker image"
	@echo "  run      - Start ChromaDB container"
	@echo "  stop     - Stop and remove ChromaDB container"
	@echo "  restart  - Restart ChromaDB container"
	@echo "  logs     - Show ChromaDB logs"
	@echo "  clean    - Remove ChromaDB container and volumes"
	@echo "  status   - Show ChromaDB container status"
	@echo "  health   - Check ChromaDB health"
	@echo "  network  - Create Docker network (if not exists)"

network:
	@echo "Creating Docker network..."
	@docker network create $(NETWORK_NAME) 2>/dev/null || echo "Network $(NETWORK_NAME) already exists"

build:
	@echo "Building ChromaDB..."
	@docker compose build

run: network
	@echo "Starting ChromaDB container..."
	@docker compose up -d
	@echo "Waiting for ChromaDB to be ready..."
	@sleep 5
	@$(MAKE) health

stop:
	@echo "Stopping ChromaDB container..."
	@docker compose down
	@echo "ChromaDB stopped"

restart: stop run

logs:
	@docker compose logs -f

clean:
	@echo "Removing ChromaDB container and volumes..."
	@docker compose down -v
	@echo "ChromaDB cleanup complete"

status:
	@echo "=== ChromaDB Container Status ==="
	@docker compose ps
	@echo ""
	@echo "=== Volume Info ==="
	@docker volume ls | grep $(VOLUME_NAME) || echo "Volume not found"

health:
	@echo "=== Testing ChromaDB Connection ==="
	@curl -s http://localhost:$(CHROMADB_PORT)/api/v1/heartbeat && \
		echo "✓ ChromaDB is healthy" || \
		echo "✗ ChromaDB is not responding"
	@echo ""
	@echo "ChromaDB API: http://localhost:$(CHROMADB_PORT)"
	@echo "Heartbeat: http://localhost:$(CHROMADB_PORT)/api/v1/heartbeat"
