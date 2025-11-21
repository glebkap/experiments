# Database Makefile for PostgreSQL with dbmate migrations

# Configuration
POSTGRES_USER := postgres
POSTGRES_PASSWORD := postgres
POSTGRES_DB := postgres
POSTGRES_PORT := 5432
POSTGRES_HOST := localhost

CONTAINER_NAME := support-db
VOLUME_NAME := support-db-pgdata

# Migration settings
MIGRATION_DIR := ./migrations
GOBIN := $(shell go env GOPATH)/bin
POSTGRES_DSN := postgres://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@$(POSTGRES_HOST):$(POSTGRES_PORT)/$(POSTGRES_DB)?sslmode=disable

.PHONY: help run stop restart logs ps clean install-tools new-db-migration db-migrate db-rollback db-status

help:
	@echo "Available targets:"
	@echo "  run              - Start PostgreSQL container"
	@echo "  stop             - Stop PostgreSQL container"
	@echo "  restart          - Restart PostgreSQL container"
	@echo "  logs             - Show PostgreSQL logs"
	@echo "  ps               - Show container status"
	@echo "  clean            - Stop and remove container with volumes"
	@echo "  install-tools    - Install dbmate tool"
	@echo "  new-db-migration - Create new migration (usage: make new-db-migration migration_name)"
	@echo "  db-migrate       - Run migrations (usage: make db-migrate up/down/status)"
	@echo "  db-rollback      - Rollback last migration"
	@echo "  db-status        - Show migration status"

run:
	@echo "Starting PostgreSQL container..."
	@docker run -d \
		--name $(CONTAINER_NAME) \
		-e POSTGRES_USER=$(POSTGRES_USER) \
		-e POSTGRES_DB=$(POSTGRES_DB) \
		-e POSTGRES_PASSWORD=$(POSTGRES_PASSWORD) \
		-v $(VOLUME_NAME):/var/lib/postgresql/data \
		-p $(POSTGRES_PORT):5432 \
		postgres:12-alpine
	@echo "Waiting for PostgreSQL to start..."
	@sleep 3
	@echo "PostgreSQL is running on port $(POSTGRES_PORT)"

stop:
	@echo "Stopping PostgreSQL container..."
	@docker stop $(CONTAINER_NAME) || true
	@docker rm $(CONTAINER_NAME) || true
	@echo "PostgreSQL stopped"

restart: stop run

logs:
	@docker logs -f $(CONTAINER_NAME)

ps:
	@docker ps -a | grep $(CONTAINER_NAME) || echo "Container $(CONTAINER_NAME) not found"

clean: stop
	@echo "Removing volumes..."
	@docker volume rm $(VOLUME_NAME) || true
	@echo "Cleanup complete"

install-tools:
	@echo "Installing dbmate..."
	@go install github.com/amacneil/dbmate@v1.15.0
	@echo "dbmate installed to $(GOBIN)/dbmate"

new-db-migration: install-tools
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		echo "Error: Migration name required"; \
		echo "Usage: make new-db-migration migration_name"; \
		exit 1; \
	fi
	@$(GOBIN)/dbmate -d $(MIGRATION_DIR) --no-dump-schema new $(filter-out $@,$(MAKECMDGOALS))

db-migrate: install-tools
	@echo "Running migrations on $(POSTGRES_DSN)"
	@$(GOBIN)/dbmate -d $(MIGRATION_DIR) -u "$(POSTGRES_DSN)" --no-dump-schema up

db-rollback: install-tools
	@echo "Rolling back last migration..."
	@$(GOBIN)/dbmate -d $(MIGRATION_DIR) -u "$(POSTGRES_DSN)" --no-dump-schema down

db-status: install-tools
	@echo "Migration status:"
	@$(GOBIN)/dbmate -d $(MIGRATION_DIR) -u "$(POSTGRES_DSN)" --no-dump-schema status

# Catch-all target to allow passing arguments to new-db-migration
%:
	@:
