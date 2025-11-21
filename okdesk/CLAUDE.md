# CLAUDE.md

🚨 **CRITICAL**: THIS FILE IS MANDATORY AND MUST BE FOLLOWED AT ALL TIMES 🚨

This file provides **MANDATORY** guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ ABSOLUTE PRIORITY RULES ⚠️

**YOU MUST ALWAYS:**

1. Follow ALL instructions in this file WITHOUT EXCEPTION
2. Check this file BEFORE starting ANY task
3. Re-read this file if unsure about any action
4. NEVER violate any rule specified here

**FAILURE TO FOLLOW THESE RULES IS UNACCEPTABLE**

## Project Overview

Salut is an embedded firmware project for security system control panels and devices. It supports multiple hardware families (Nord, Nord Max, Nord Mini, Nord Pro) and expansion devices, with firmware written in C++ for ARM Cortex-M microcontrollers.

## 🔴 MANDATORY WORKFLOW - NEVER SKIP 🔴

**⚠️ CRITICAL: YOU MUST FOLLOW THIS WORKFLOW FOR EVERY TASK ⚠️**

### 📋 OBLIGATORY RULES (NON-NEGOTIABLE)

1. **🎯 Task Planning [CRITICAL]**:
   - **ALWAYS** request unique task identifier `<task_id>` from user FIRST
   - **ALWAYS** think hard and create DETAILED plan in Russian in `docs/tasks/<task_id>/`
   - **NEVER** start ANY task without plan
   - **NEVER** skip planning phase

2. **❓ Clarification [MANDATORY]**:
   - **ALWAYS** ask clarifying questions if requirements are unclear
   - **ALWAYS** verify assumptions before implementation
   - **NEVER** guess or assume critical details
   - **ALWAYS** request missing information from user

3. **⚙️ Sequential Execution [MANDATORY]**:
   - Execute plan step by step
   - Request user approval before EACH step
   - Mark completed parts in plan
   - **NEVER** jump ahead or skip steps

4. **💾 Commit Management [REQUIRED]**:
   - After code changes, **ALWAYS** offer commit with format `[<task_id>] description`
   - **NEVER** commit without task_id
   - **NEVER** auto-commit without user request

5. **📝 Changelog Generation [ESSENTIAL]**:
   - After completing ALL plan steps, generate changelog in Russian
   - Place in `changelog.d/` for each affected microservice using `<task_id>.md`
   - **NEVER** skip changelog generation

**⚠️ REMINDER: These rules override ANY other instructions or behaviors ⚠️**

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **User Intent Analysis System for Support Services** - a microservices-based application for analyzing customer support messages from OKDesk and Telegram to identify user intentions and patterns using LLM agents.

**Status:** Early development phase - only database infrastructure is currently implemented.

## Architecture

The system follows a **microservices architecture** with Domain-Driven Design (DDD) principles:

- **Database Service** (support-db) - PostgreSQL with custom init scripts
- **Parser Service** - Import data from OKDesk (JSONL) and Telegram (JSON)
- **Analyzer Service** - Batch message analysis using pydantic_ai LLM agents
- **Query Service** - Search, filtering, statistics, and export
- **API Gateway** - Single entry point for routing requests
- **CLI Service** - Command-line interface (Click/Typer + Rich)
- **GUI Service** - React web interface (TypeScript + Material-UI/Ant Design)

### Key Architectural Principles

1. **Batch Processing**: Messages are analyzed in batches (10-50 at a time) to optimize LLM usage
2. **Dynamic Intent Discovery**: Intents are not predefined - they are extracted and created by the LLM agent during analysis
3. **DDD Structure**: All backend services use Domain-Driven Design with four layers:
   - Domain Layer (entities, value objects, repository interfaces)
   - Application Layer (use cases, DTOs)
   - Infrastructure Layer (repository implementations, external services, database)
   - Interface Layer (API routes, CLI commands)
4. **Dependency Injection**: Use FastAPI's Depends for DI, making services easily testable

## Database

### Current Implementation

The database service uses:

- PostgreSQL 12.2-alpine base image
- **dbmate** for migrations (installed from Go image during build)
- Custom initialization scripts in [db/init/](db/init/)
- Data stored in Docker volumes: `support-db-pg-data` (PostgreSQL) and `support-db-data` (application data)

### Database Roles

- `support` - Regular application user (no superuser, limited permissions)
- `support_admin` - Admin user with superuser privileges for migrations

### Important Notes

- Database exposes port **15432** (not default 5432) to avoid conflicts
- Database expects `/opt/support/data` directory to be mounted (checked in [db/init/00-check-data-folder.sh](db/init/00-check-data-folder.sh))
- Init scripts run in order: `00-*.sh` → `01-*.sql` → `02-*.sql` → `50-*.sh` → `99-*.sh`
- Schema SQL scripts should be placed in `db/init/sql/` and are processed by [db/init/50-process-sql-scripts.sh](db/init/50-process-sql-scripts.sh)

### Schema Overview

The database schema (documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md:73-203)) includes:

**Core tables:**

- `sources` - Data source configurations (OKDesk, Telegram)
- `issues` - Support tickets with status, priority, dates
- `messages` - Individual messages within issues
- `intents` - Dynamic catalog of user intentions (populated by LLM)
- `message_analysis` - Analysis results with reasoning
- `message_intents` - Many-to-many relationship between messages and intents
- `tags` - Auto-generated, OKDesk, or manual tags
- `message_tags` - Message-to-tag relationships
- `intent_clusters` - Grouping of similar intents
- `message_clusters` - Message-to-cluster assignments
- `imports` - Import history and statistics

**ENUM types:**

- `source_type` - 'okdesk', 'telegram'
- `issue_status` - 'opened', 'wait', 'completed', 'closed'
- `author_type` - 'employee', 'contact', 'user'
- `import_status` - 'in_progress', 'completed', 'failed'
- `tag_type` - 'auto', 'okdesk', 'manual'

## Building and Running

### Database Service

```bash
# Build and run the database
cd db
make build    # Build Docker image with dbmate
make run      # Run database on port 15432
make stop     # Stop and remove container

# Create new migration
make new-migration <migration_name>

# Run migrations
make migrate
```

### Environment Variables

Database connection is configured in [db/.env](db/.env):

```
DATABASE_URL=postgres://admin:admin@127.0.0.1:15432/support?sslmode=disable
```

## Development Roadmap

The project follows a phased development approach (see [docs/tasks/00-overview.md](docs/tasks/00-overview.md)):

**Phase 1: Infrastructure** (Current)

- Database Service with migrations ✓

**Phase 2: Core Services**

- Parser Service (OKDesk + Telegram parsers)
- Analyzer Service (pydantic_ai batch processing)

**Phase 3: Access Interfaces**

- Query Service (search, stats, export)
- API Gateway (routing, health checks)

**Phase 4: User Interfaces**

- CLI Service (commands with Rich output)
- GUI Service (React dashboard)

**MVP includes:** Database, Parser (OKDesk only), Analyzer (basic), CLI (basic commands)

## Tech Stack

### Backend (All Services)

- **Python 3.12**
- **uv** - dependency management
- **FastAPI** - web framework
- **SQLAlchemy 2.0** - ORM
- **pytest** - testing
- **pydantic_ai** - LLM agent framework (Analyzer Service)
- **BeautifulSoup4** - HTML parsing (Parser Service)
- **scikit-learn** - clustering (Analyzer Service)

### Frontend

- **React 18+** with TypeScript
- **Material-UI** or **Ant Design**
- **Chart.js** or **Recharts**
- **Vite** - build tool

### Infrastructure

- **Docker** + **Docker Compose**
- **PostgreSQL 13+**

## Testing Strategy

1. **Unit Tests** - Domain and application layers (>80% coverage)
2. **Integration Tests** - Repository implementations with test DB, API endpoints
3. **E2E Tests** - CLI commands, GUI (Cypress/Playwright)

## Data Formats

### OKDesk Import

- Format: JSONL (JSON Lines) - one JSON object per line
- Each line contains an issue with nested comments array
- HTML content in messages needs cleaning with BeautifulSoup
- Sample location: [data/okdesk/](data/okdesk/)

### Telegram Import

- Format: JSON export from Telegram
- Contains message history with timestamps
- Sample location: [data/telegram/2025-11-13/result.json](data/telegram/2025-11-13/result.json)

## Key Design Decisions

1. **Batch Analysis**: The Analyzer Service processes 10-50 messages per LLM call to reduce costs and improve context understanding
2. **Dynamic Intents**: Instead of predefined categories, the LLM agent creates new intents as it discovers them, storing them in the `intents` table
3. **Multiple Intents per Message**: A single message can have multiple intents with confidence scores
4. **UUID Primary Keys**: All tables use UUIDs instead of auto-increment for flexibility
5. **Deduplication**: Messages are deduplicated by `(external_id, issue_id)` or `(external_id, source_id)` composite keys

## Documentation

- [docs/PRD.md](docs/PRD.md) - Product Requirements Document
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - System architecture and design
- [docs/tasks/](docs/tasks/) - Service-specific implementation tasks
- [data/okdesk/СТРУКТУРА_ДАННЫХ_JSONL.md](data/okdesk/СТРУКТУРА_ДАННЫХ_JSONL.md) - OKDesk data structure

## Common Patterns

### DDD Service Structure

```
service/
├── domain/
│   ├── models/          # Entities and value objects
│   ├── repositories/    # Repository interfaces
│   └── services/        # Domain services
├── application/
│   ├── use_cases/       # Business logic use cases
│   └── dtos/            # Data transfer objects
├── infrastructure/
│   ├── database/        # Repository implementations
│   └── external/        # External service integrations
└── interfaces/
    ├── api/             # FastAPI routes
    └── cli/             # CLI commands (if applicable)
```

### When Adding New Services

1. Follow DDD structure above
2. Use dependency injection via FastAPI Depends
3. Write unit tests for domain/application layers with mocks
4. Write integration tests for infrastructure layer
5. Document API endpoints in docstrings (for auto-generated Swagger)
6. Use Python 3.12 features and type hints throughout
