# Changelog: Parser Service

**Task ID:** 02-parser-service  
**Дата:** 21.11.2025  
**Тип:** Feature

## Изменения

### Добавлено

**Domain Layer:**
- Созданы domain models: `Source`, `Issue`, `Message`, `ImportJob`
- Созданы ENUMs: `SourceType`, `IssueStatus`, `AuthorType`, `ImportStatus`
- Созданы repository interfaces для всех сущностей
- Реализованы domain services: `DeduplicationService`, `ImportService`

**Infrastructure Layer:**
- SQLAlchemy модели с маппингом на БД
- Async database connection через asyncpg
- PostgreSQL repository implementations (4 класса)
- OKDesk parser (JSONL) с очисткой HTML
- Telegram parser (JSON export)
- HTTP client для Analyzer Service с retry logic

**Application Layer:**
- DTOs для import operations
- Use case `ImportOKDeskUseCase`

**Interface Layer:**
- FastAPI routes (`/api/v1/import/okdesk`, `/health`)
- Pydantic schemas для request/response
- Main приложение с CORS middleware

**Infrastructure:**
- Dockerfile с multi-stage build
- uv dependency management
- Configuration через Pydantic Settings

### Технологии

- Python 3.12
- FastAPI + uvicorn
- SQLAlchemy 2.0 (async)
- asyncpg
- BeautifulSoup4 + lxml
- httpx
- pytest

## Файлы

- `services/parser/src/domain/` - 14 файлов
- `services/parser/src/infrastructure/` - 12 файлов
- `services/parser/src/application/` - 3 файла
- `services/parser/src/interfaces/` - 3 файла
- `services/parser/pyproject.toml`
- `services/parser/Dockerfile`
- `services/parser/README.md`

**Всего:** ~35 Python файлов

## Статус

✅ MVP готов к запуску (требует тестирование)

## Следующие шаги

1. Написать unit и integration тесты
2. Полная реализация DI через dependencies.py
3. Добавить Telegram import use case
4. Интеграционное тестирование с Database Service
