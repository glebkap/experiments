# Parser Service

Микросервис для импорта данных из OKDesk и Telegram.

## Архитектура

Использует **DDD (Domain-Driven Design)** подход:

- **Domain Layer**: Бизнес-логика, сущности, repository interfaces
- **Application Layer**: Use cases, DTOs
- **Infrastructure Layer**: Реализация репозиториев, парсеры, HTTP клиенты
- **Interface Layer**: FastAPI API endpoints

## Установка и запуск

### Локально с uv

```bash
# Установить зависимости
uv sync

# Запустить сервис
uv run python -m uvicorn src.main:app --reload --port 8001
```

### Docker

```bash
# Собрать образ
docker build -t support-parser .

# Запустить контейнер
docker run -p 8001:8001 support-parser
```

## Импорт данных

### Импорт файла OKDesk (JSONL)

```bash
cd services/parser

# Запустить импорт
uv run python scripts/import_okdesk.py ../../data/okdesk/2025-11-14.jsonl

# С указанием source_id
uv run python scripts/import_okdesk.py ../../data/okdesk/2025-11-14.jsonl <source_id>
```

**Примечание:** Убедитесь, что база данных запущена:
```bash
cd ../../db
make run
```

## API Endpoints

- `POST /api/v1/import/okdesk` - импорт файла OKDesk (JSONL)
- `GET /health` - health check

## Конфигурация

Создайте `.env` файл на основе `.env.example`:

```bash
cp .env.example .env
```

## Технологии

- Python 3.12
- FastAPI
- SQLAlchemy 2.0 (async)
- PostgreSQL (asyncpg)
- BeautifulSoup4
- httpx
