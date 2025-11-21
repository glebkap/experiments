# Query Service - Декомпозиция задач

**Сервис:** support-query
**Приоритет:** Средний
**Технологии:** Python 3.12, FastAPI, SQLAlchemy, uv
**Архитектура:** DDD (Domain-Driven Design)

---

## Обзор

Query Service предоставляет API для поиска, фильтрации, статистики и экспорта данных. Только чтение из БД, без изменения данных.

---

## DDD Структура

```
services/query/
├── src/
│   ├── domain/
│   │   ├── models/          # Read models (упрощенные)
│   │   │   ├── issue.py
│   │   │   ├── message.py
│   │   │   ├── intent.py
│   │   │   └── stats.py
│   │   └── repositories/
│   │       ├── search_repository.py
│   │       ├── stats_repository.py
│   │       └── export_repository.py
│   ├── application/
│   │   ├── use_cases/
│   │   │   ├── search_messages.py
│   │   │   ├── get_issue_details.py
│   │   │   ├── get_statistics.py
│   │   │   ├── get_timeline.py
│   │   │   └── export_data.py
│   │   └── dto/
│   │       ├── search_dto.py
│   │       ├── stats_dto.py
│   │       └── export_dto.py
│   ├── infrastructure/
│   │   ├── persistence/
│   │   │   ├── postgres/
│   │   │   │   ├── search_repository_impl.py
│   │   │   │   ├── stats_repository_impl.py
│   │   │   │   └── export_repository_impl.py
│   │   │   ├── models.py
│   │   │   └── database.py
│   │   └── export/
│   │       ├── csv_exporter.py
│   │       └── json_exporter.py
│   ├── interfaces/
│   │   └── api/
│   │       ├── routes.py
│   │       └── schemas.py
│   ├── config.py
│   ├── dependencies.py
│   └── main.py
├── tests/
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## Задачи

### 1. Настройка проекта

- [ ] Создать структуру директорий
- [ ] Инициализировать uv проект
- [ ] Добавить зависимости: fastapi, uvicorn, sqlalchemy, psycopg2-binary
- [ ] Создать Dockerfile

### 2. Domain Layer

#### 2.1 Domain Models (Read Models)

- [ ] Создать упрощенные read-only модели
  - Issue (без полей для update)
  - Message с analysis результатами
  - Intent
  - Tag
  - Stats models (IntentStats, TagStats, TimelinePoint)

#### 2.2 Repository Interfaces

- [ ] SearchRepository
  - Методы: full_text_search, filter_by_intents, filter_by_tags, filter_by_dates

- [ ] StatsRepository
  - Методы: get_intent_stats, get_tag_stats, get_source_stats, get_timeline

- [ ] ExportRepository
  - Методы: export_messages, export_issues

### 3. Application Layer

#### 3.1 DTOs

- [ ] SearchRequest, SearchResponse
- [ ] IssueDetailsResponse
- [ ] StatsRequest, StatsResponse
- [ ] TimelineRequest, TimelineResponse
- [ ] ExportRequest, ExportResponse

#### 3.2 Use Cases

- [ ] SearchMessagesUseCase
  - Полнотекстовый поиск с фильтрами
  - Пагинация

- [ ] GetIssueDetailsUseCase
  - Получить issue со всеми messages и analysis

- [ ] GetStatisticsUseCase
  - Агрегация по намерениям/тегам/источникам

- [ ] GetTimelineUseCase
  - Временная динамика с группировкой (день/неделя/месяц)

- [ ] ExportDataUseCase
  - Применение фильтров
  - Вызов экспортера (CSV/JSON)

### 4. Infrastructure Layer

#### 4.1 Persistence

- [ ] Создать SQLAlchemy модели (read-only views/queries)
- [ ] Реализовать SearchRepositoryImpl
  - Использовать pg_trgm для полнотекстового поиска
  - Joins для фильтрации по intents/tags
  - Pagination

- [ ] Реализовать StatsRepositoryImpl
  - SQL агрегация (COUNT, GROUP BY)
  - Joins для связанных таблиц

- [ ] Реализовать ExportRepositoryImpl
  - Query builder с фильтрами

#### 4.2 Exporters

- [ ] CSVExporter
  - Использовать csv module
  - Стриминг для больших выборок

- [ ] JSONExporter
  - Сериализация в JSON
  - Форматирование

### 5. Interface Layer (API)

#### 5.1 Routes

- [ ] GET /api/v1/search - полнотекстовый поиск
- [ ] GET /api/v1/issues - список обращений
- [ ] GET /api/v1/issues/{id} - детали обращения
- [ ] GET /api/v1/stats/intents - статистика по намерениям
- [ ] GET /api/v1/stats/tags - статистика по тегам
- [ ] GET /api/v1/stats/sources - статистика по источникам
- [ ] GET /api/v1/stats/timeline - временная статистика
- [ ] GET /api/v1/clusters - список кластеров
- [ ] GET /api/v1/clusters/{id}/messages - сообщения в кластере
- [ ] POST /api/v1/export - экспорт данных

### 6. Configuration and DI

- [ ] Создать config.py
- [ ] Создать dependencies.py с функциями для DI
- [ ] Создать main.py с FastAPI app

### 7. Тестирование

- [ ] Unit тесты для use cases
- [ ] Интеграционные тесты с БД
- [ ] Тесты для экспортеров

### 8. Документация

- [ ] README.md с описанием API
- [ ] Swagger/OpenAPI документация

---

## Критерии готовности

- [ ] Все endpoints работают
- [ ] Полнотекстовый поиск работает
- [ ] Фильтрация корректна
- [ ] Статистика считается правильно
- [ ] Экспорт в CSV и JSON работает
- [ ] Тесты проходят

---

## Зависимости

**Требует:**
- Database Service готов
- Analyzer Service (для наличия данных анализа)

**Блокирует:**
- CLI Service (для команд поиска/статистики)
- GUI Service (для отображения данных)
