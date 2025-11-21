# Parser Service - Декомпозиция задач

**Сервис:** support-parser
**Приоритет:** Высокий (блокирует Analyzer)
**Технологии:** Python 3.12, FastAPI, psycopg2, BeautifulSoup4, uv
**Архитектура:** DDD (Domain-Driven Design)

---

## Обзор

Parser Service отвечает за импорт данных из различных источников (OKDesk, Telegram) в базу данных. Использует DDD подход для организации бизнес-логики.

---

## DDD Структура

```
services/parser/
├── src/
│   ├── domain/              # Domain layer (бизнес-логика)
│   │   ├── models/          # Domain models (entities, value objects)
│   │   │   ├── source.py
│   │   │   ├── issue.py
│   │   │   ├── message.py
│   │   │   └── import_job.py
│   │   ├── repositories/    # Repository interfaces
│   │   │   ├── source_repository.py
│   │   │   ├── issue_repository.py
│   │   │   ├── message_repository.py
│   │   │   └── import_repository.py
│   │   └── services/        # Domain services
│   │       ├── deduplication_service.py
│   │       └── import_service.py
│   ├── application/         # Application layer (use cases)
│   │   ├── use_cases/
│   │   │   ├── import_okdesk.py
│   │   │   ├── import_telegram.py
│   │   │   └── get_import_status.py
│   │   └── dto/             # Data Transfer Objects
│   │       └── import_dto.py
│   ├── infrastructure/      # Infrastructure layer
│   │   ├── persistence/     # Repository implementations
│   │   │   ├── postgres/
│   │   │   │   ├── source_repository_impl.py
│   │   │   │   ├── issue_repository_impl.py
│   │   │   │   ├── message_repository_impl.py
│   │   │   │   └── import_repository_impl.py
│   │   │   ├── models.py    # SQLAlchemy models
│   │   │   └── database.py  # DB connection
│   │   ├── parsers/         # External file parsers
│   │   │   ├── okdesk_parser.py
│   │   │   └── telegram_parser.py
│   │   └── http/            # HTTP clients
│   │       └── analyzer_client.py
│   ├── interfaces/          # Interface adapters (API)
│   │   ├── api/
│   │   │   ├── routes.py
│   │   │   └── schemas.py   # Pydantic request/response models
│   │   └── cli/             # CLI interface (если нужно)
│   ├── config.py
│   ├── dependencies.py      # Dependency injection
│   └── main.py
├── tests/
│   ├── unit/
│   │   ├── domain/
│   │   └── application/
│   ├── integration/
│   │   └── test_import.py
│   └── fixtures/
│       ├── okdesk_sample.jsonl
│       └── telegram_sample.json
├── Dockerfile
├── pyproject.toml           # uv project file
├── uv.lock
└── README.md
```

---

## Задачи

### 1. Настройка проекта

- [ ] Создать структуру директорий согласно DDD
- [ ] Инициализировать uv проект
  ```bash
  cd services/parser
  uv init
  ```

- [ ] Добавить зависимости через uv
  ```bash
  uv add fastapi uvicorn psycopg2-binary sqlalchemy beautifulsoup4
  uv add python-multipart httpx pydantic pydantic-settings
  uv add --dev pytest pytest-asyncio
  ```

- [ ] Создать `pyproject.toml` с настройками
  - Настроить pytest
  - Настроить paths

- [ ] Создать Dockerfile
  - Использовать uv для установки зависимостей
  - Multi-stage build

### 2. Domain Layer

#### 2.1 Domain Models (Entities)

- [ ] Создать `domain/models/source.py`
  - Enum SourceType (okdesk, telegram)
  - Dataclass Source

- [ ] Создать `domain/models/issue.py`
  - Enum IssueStatus
  - Dataclass Issue
  - Поля: id, external_id, source_id, title, description, status, priority, timestamps

- [ ] Создать `domain/models/message.py`
  - Enum AuthorType
  - Dataclass Message
  - Поля: id, issue_id, external_id, author info, content, is_public, published_at

- [ ] Создать `domain/models/import_job.py`
  - Enum ImportStatus
  - Dataclass ImportJob
  - Поля: id, source_id, filename, status, stats, timestamps, error_message

#### 2.2 Repository Interfaces

- [ ] Создать `domain/repositories/source_repository.py`
  - Abstract базовый класс
  - Методы: get_by_id, get_by_type, create

- [ ] Создать `domain/repositories/issue_repository.py`
  - Abstract базовый класс
  - Методы: get_by_external_id, create, update

- [ ] Создать `domain/repositories/message_repository.py`
  - Abstract базовый класс
  - Методы: get_by_external_id, create, update, bulk_create

- [ ] Создать `domain/repositories/import_repository.py`
  - Abstract базовый класс
  - Методы: create, update, get_by_id, list_all

#### 2.3 Domain Services

- [ ] Создать `domain/services/deduplication_service.py`
  - Зависимости: issue_repo, message_repo
  - Методы:
    - is_issue_duplicate(external_id, source_id) -> bool
    - is_message_duplicate(external_id, issue_id) -> bool

- [ ] Создать `domain/services/import_service.py`
  - Координация процесса импорта
  - Управление транзакциями
  - Логирование статистики

### 3. Application Layer (Use Cases)

#### 3.1 DTOs

- [ ] Создать `application/dto/import_dto.py`
  - ImportRequest
  - ImportResponse
  - ImportStats
  - ImportProgress

#### 3.2 Use Cases

- [ ] Создать `application/use_cases/import_okdesk.py`
  - Класс ImportOKDeskUseCase
  - Зависимости через конструктор (DI)
  - Метод execute(file_path, source_id) -> import_id
  - Алгоритм:
    1. Создать import job
    2. Парсить файл построчно
    3. Для каждой строки:
       - Проверить дубликаты
       - Сохранить issue
       - Сохранить messages
       - Собрать message_ids
    4. Обновить import job status
    5. Вызвать analyzer
    6. Вернуть import_id

- [ ] Создать `application/use_cases/import_telegram.py`
  - Аналогично ImportOKDeskUseCase

- [ ] Создать `application/use_cases/get_import_status.py`
  - Класс GetImportStatusUseCase
  - Метод execute(import_id) -> ImportStatus

- [ ] Создать `application/use_cases/list_imports.py`
  - Класс ListImportsUseCase
  - Метод execute(limit, offset, filters) -> List[ImportJob]

### 4. Infrastructure Layer

#### 4.1 Persistence

- [ ] Создать `infrastructure/persistence/models.py`
  - SQLAlchemy модели для всех таблиц:
    - SourceModel
    - IssueModel
    - MessageModel
    - ImportModel
  - Маппинг на domain модели

- [ ] Создать `infrastructure/persistence/database.py`
  - Настройка engine
  - SessionLocal factory
  - Dependency для получения session

- [ ] Создать `infrastructure/persistence/postgres/source_repository_impl.py`
  - Реализация SourceRepository
  - Использует SQLAlchemy session
  - Методы: get_by_id, get_by_type, create

- [ ] Создать `infrastructure/persistence/postgres/issue_repository_impl.py`
  - Реализация IssueRepository
  - Методы: get_by_external_id, create, update

- [ ] Создать `infrastructure/persistence/postgres/message_repository_impl.py`
  - Реализация MessageRepository
  - Методы: get_by_external_id, create, update, bulk_create

- [ ] Создать `infrastructure/persistence/postgres/import_repository_impl.py`
  - Реализация ImportRepository
  - Методы: create, update, get_by_id, list_all

#### 4.2 Parsers

- [ ] Создать `infrastructure/parsers/okdesk_parser.py`
  - Класс OKDeskParser
  - Метод parse_file(file_path) -> Iterator[dict]
    - Читает JSONL построчно
    - Парсит JSON
    - Yield каждое обращение
  - Метод extract_issue(data) -> dict
    - Извлекает поля issue из JSON
  - Метод extract_comments(data) -> list[dict]
    - Извлекает комментарии

- [ ] Создать `infrastructure/parsers/telegram_parser.py`
  - Класс TelegramParser
  - Метод parse_file(file_path) -> dict
    - Читает JSON файл
    - Парсит Telegram Export структуру
  - Метод extract_messages(data) -> list[dict]
    - Извлекает сообщения из диалога

#### 4.3 HTTP Clients

- [ ] Создать `infrastructure/http/analyzer_client.py`
  - Класс AnalyzerClient
  - Зависимость: base_url из config
  - Метод analyze_batch(message_ids, batch_size) -> response
    - POST /api/v1/analyze/batch
    - Обработка ошибок
    - Retry логика

### 5. Interface Layer (API)

#### 5.1 Schemas

- [ ] Создать `interfaces/api/schemas.py`
  - Pydantic модели для request/response:
    - ImportResponse
    - ImportStatusResponse
    - ImportListResponse
    - ErrorResponse

#### 5.2 Routes

- [ ] Создать `interfaces/api/routes.py`
  - POST /api/v1/import/okdesk
    - Прием файла (multipart/form-data)
    - Сохранение во временную директорию
    - Вызов use case
    - Возврат import_id и статуса
  - POST /api/v1/import/telegram
    - Аналогично
  - GET /api/v1/import/{id}
    - Получение статуса импорта
  - GET /api/v1/imports
    - Список импортов с пагинацией

### 6. Configuration and DI

- [ ] Создать `config.py`
  - Класс Settings (Pydantic BaseSettings)
  - Переменные:
    - DB connection params
    - Analyzer service URL
    - Temp directory path
  - Загрузка из .env

- [ ] Создать `dependencies.py`
  - Функции для DI:
    - get_settings()
    - get_db()
    - get_source_repository()
    - get_issue_repository()
    - get_message_repository()
    - get_import_repository()
    - get_deduplication_service()
    - get_okdesk_parser()
    - get_telegram_parser()
    - get_analyzer_client()
    - get_import_okdesk_use_case()
    - get_import_telegram_use_case()

- [ ] Создать `main.py`
  - FastAPI app
  - Include routers
  - Health check endpoint
  - CORS middleware
  - Exception handlers

### 7. Тестирование

- [ ] Создать unit тесты для domain layer
  - tests/unit/domain/test_deduplication_service.py
  - tests/unit/domain/test_import_service.py

- [ ] Создать unit тесты для application layer
  - tests/unit/application/test_import_okdesk_use_case.py
  - Использовать моки для repositories

- [ ] Создать интеграционные тесты
  - tests/integration/test_import_okdesk.py
  - Использовать тестовую БД
  - Проверить импорт реального файла

- [ ] Создать тестовые fixtures
  - fixtures/okdesk_sample.jsonl (несколько обращений)
  - fixtures/telegram_sample.json

### 8. Документация

- [ ] Создать README.md
  - Описание сервиса
  - Архитектура DDD (слои)
  - Установка и запуск
  - API документация
  - Примеры использования

- [ ] Добавить docstrings ко всем классам и методам

---

## Критерии готовности

- [ ] Все слои DDD реализованы
- [ ] Dependency Injection настроен
- [ ] API endpoints работают
- [ ] Импорт OKDesk файла работает
- [ ] Импорт Telegram файла работает
- [ ] Дедупликация работает корректно
- [ ] Вызов Analyzer Service работает
- [ ] Unit тесты покрывают domain и application layers
- [ ] Интеграционные тесты проходят
- [ ] Документация актуальна

---

## Зависимости

**Требует:**
- Database Service готов

**Блокирует:**
- Analyzer Service (частично)
- CLI Service (импорт команды)

**Интегрируется с:**
- Analyzer Service (HTTP вызовы)
