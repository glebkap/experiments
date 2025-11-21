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

### Просмотр статистики

```bash
cd services/parser

# Показать статистику по базе данных
uv run python scripts/show_stats.py
```

### Очистка базы данных

```bash
cd services/parser

# Удалить все данные из всех таблиц (с подтверждением)
uv run python scripts/clear_db.py
```

**Примечание:** Убедитесь, что база данных запущена:

```bash
cd ../../db
make run
```

## API Endpoints

### Import

- `POST /api/v1/import/okdesk` - импорт файла OKDesk (JSONL)

### Statistics

- `GET /api/v1/stats` - общая статистика по всем сущностям
- `GET /api/v1/stats/issues` - статистика по тикетам (по статусам, приоритетам, источникам)
- `GET /api/v1/stats/messages` - статистика по сообщениям (по типу автора, публичность)
- `GET /api/v1/stats/imports` - статистика по импортам (по статусам, последний импорт)

### Health

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
