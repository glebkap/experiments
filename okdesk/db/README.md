# Database Service

PostgreSQL database for the User Intent Analysis System for Support Services.

## Overview

This service provides the core database infrastructure using:
- **PostgreSQL 12-alpine** - Lightweight PostgreSQL image
- **dbmate** - Database migration tool
- **Docker** - Containerization

## Quick Start

```bash
# Start the database
make run

# Run migrations
make db-migrate

# Check migration status
make db-status

# Stop the database
make stop

# Clean everything (stop + remove volumes)
make clean
```

## Database Configuration

The database runs with the following default settings (configured in `Makefile`):

- **Host**: localhost
- **Port**: 5432
- **Database**: postgres
- **User**: postgres
- **Password**: postgres

You can override these by editing the `Makefile` or `.env` file.

## Available Make Targets

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make run` | Start PostgreSQL container |
| `make stop` | Stop PostgreSQL container |
| `make restart` | Restart PostgreSQL container |
| `make logs` | Show PostgreSQL logs |
| `make ps` | Show container status |
| `make clean` | Stop and remove container with volumes |
| `make install-tools` | Install dbmate migration tool |
| `make new-db-migration <name>` | Create new migration |
| `make db-migrate` | Run all pending migrations |
| `make db-rollback` | Rollback last migration |
| `make db-status` | Show migration status |

## Migrations

Migrations are stored in `migrations/` and managed by `dbmate`.

### Creating a New Migration

```bash
make new-db-migration add_user_table
```

This creates a timestamped migration file like `20251121124809_add_user_table.sql`:

```sql
-- migrate:up
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL
);

-- migrate:down
DROP TABLE IF EXISTS users;
```

### Applying Migrations

```bash
make db-migrate
```

### Rolling Back

```bash
make db-rollback
```

## Database Schema

The database contains the following components:

### ENUM Types

- `source_type` - 'okdesk', 'telegram'
- `issue_status` - 'opened', 'wait', 'completed', 'closed'
- `author_type` - 'employee', 'contact', 'user'
- `import_status` - 'in_progress', 'completed', 'failed'
- `tag_type` - 'auto', 'okdesk', 'manual'

### Core Tables

1. **sources** - Data source configurations (OKDesk, Telegram)
2. **issues** - Support tickets/issues
3. **messages** - Individual messages within issues
4. **intents** - Dynamic catalog of user intentions (populated by LLM)
5. **message_analysis** - Analysis results for messages
6. **message_intents** - Many-to-many relationship between messages and intents
7. **tags** - Auto-generated, OKDesk, or manual tags
8. **message_tags** - Message-to-tag relationships
9. **intent_clusters** - Grouping of similar intents
10. **message_clusters** - Message-to-cluster assignments
11. **imports** - Import history and statistics

### Indexes

Performance indexes are created on frequently queried columns:
- `idx_messages_issue_id` - Messages by issue
- `idx_messages_published_at` - Temporal ordering
- `idx_message_analysis_message_id` - Analysis lookup
- `idx_issues_external_id` - Issue deduplication
- `idx_issues_created_at` - Issue ordering
- `idx_message_intents_intent_id` - Intent grouping
- `idx_message_tags_tag_id` - Tag grouping
- `idx_intents_code` - Intent lookup by code
- `idx_messages_content_trgm` - Full-text search (GIN index)

### PostgreSQL Extensions

- **pgcrypto** - Provides `gen_random_uuid()` for UUID generation
- **pg_trgm** - Trigram-based full-text search

## Connection

### From Application Code

```python
# Python example
DATABASE_URL = "postgres://postgres:postgres@localhost:5432/postgres?sslmode=disable"
```

### Using psql

```bash
docker exec -it support-db psql -U postgres -d postgres
```

Or from host:

```bash
psql -h localhost -p 5432 -U postgres -d postgres
```

## Data Persistence

Data is stored in Docker volumes:
- `support-db-pgdata` - PostgreSQL data directory

**Warning**: Running `make clean` will delete all data!

## Development Workflow

1. **Start database**: `make run`
2. **Create migration**: `make new-db-migration add_feature`
3. **Edit migration file** in `migrations/`
4. **Apply migration**: `make db-migrate`
5. **Verify**: `make db-status`

## Troubleshooting

### Port Already in Use

If port 5432 is already in use, change `POSTGRES_PORT` in `Makefile`.

### Connection Refused

Make sure the database is running:
```bash
make ps
```

If not running:
```bash
make run
```

### Migration Fails

Check the migration SQL syntax:
```bash
make logs
```

Rollback if needed:
```bash
make db-rollback
```

### Clean Start

To start fresh with empty database:
```bash
make clean
make run
make db-migrate
```

## Architecture Notes

- **UUID Primary Keys**: All tables use UUIDs for flexibility in distributed systems
- **Deduplication**: Composite unique constraints on `(external_id, source_id)` or `(external_id, issue_id)`
- **Dynamic Intents**: Intents are discovered and created by the LLM agent during analysis
- **Confidence Scores**: All intent and tag assignments include confidence values (0.0 to 1.0)
- **JSONB Config**: Source and import statistics use JSONB for flexible schema

## See Also

- [Architecture Documentation](../docs/ARCHITECTURE.md)
- [Project PRD](../docs/PRD.md)
- [Database Task](../docs/tasks/01-database-service.md)
