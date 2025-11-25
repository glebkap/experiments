# Changelog: Analyzer Service Extensions

**Task ID:** 04-analyzer-extensions
**Дата:** 25.11.2025
**Сервис:** Analyzer Service

---

## Добавлено

### Новые API Endpoints

#### Issues Endpoints
- `GET /api/v1/analyzer/issues` - Получение списка issues с фильтрацией и пагинацией
  - Фильтры: status, source_id, priority, date_from, date_to
  - Пагинация: limit, offset
- `GET /api/v1/analyzer/issues/{id}` - Детальная информация об issue с messages

#### Clusters Endpoints
- `GET /api/v1/analyzer/clusters/{id}/issues` - Получение issues в кластере (улучшенная версия)
  - Сортировка по расстоянию до центроида
  - Пагинация: limit, offset

#### Search Endpoints
- `GET /api/v1/analyzer/search/fulltext` - Полнотекстовый поиск по PostgreSQL FTS
  - Поиск в preprocessed_issues
  - Ранжирование по релевантности

#### Stats Endpoints
- `GET /api/v1/analyzer/stats/processing` - Статистика обработки
- `GET /api/v1/analyzer/stats/sources` - Статистика по источникам данных
- `GET /api/v1/analyzer/stats/clusters` - Статистика по кластерам
- `GET /api/v1/analyzer/stats/timeline` - Временная статистика (группировка по дням/неделям/месяцам)

#### Export Endpoint
- `POST /api/v1/analyzer/export` - Экспорт данных в CSV или JSON формат

### Новые Repository Interfaces

#### IssueRepository (расширение)
- `get_issues_with_filters()` - Получение issues с фильтрами
- `get_issue_with_messages()` - Получение issue с messages
- `count_issues_with_filters()` - Подсчет issues с фильтрами
- `fulltext_search()` - Полнотекстовый поиск
- `get_issue_source_info()` - Информация об источнике issue

#### ClusterRepository (расширение)
- `get_cluster_with_issues()` - Получение кластера с issues (полные объекты)
- `get_cluster_stats()` - Статистика по кластерам
- `get_issue_cluster()` - Информация о кластере для issue

#### StatsRepository (новый)
- `get_processing_stats()` - Общая статистика обработки
- `get_sources_stats()` - Статистика по источникам
- `get_timeline_stats()` - Временная статистика

### Новые DTOs
- `IssueListItemDTO`, `IssueListResponseDTO`
- `IssueDetailDTO`, `MessageDTO`
- `ProcessingStatsDTO`, `SourceStatsDTO`, `TimelinePointDTO`, `ClusterStatsDTO`
- `ExportFiltersDTO`, `ExportRequestDTO`, `ExportResultDTO`

### Новые Pydantic Schemas
- Schemas для всех новых endpoints (request/response validation)

### SQLAlchemy Models
- `SourceModel` - модель для таблицы sources
- Relationship между IssueModel и SourceModel

---

## Изменено

- `IssueRepository` interface - добавлены новые методы
- `ClusterRepository` interface - добавлены новые методы
- `IssueRepositoryImpl` - реализация новых методов
- `ClusterRepositoryImpl` - реализация новых методов
- `dependencies.py` - добавлен `get_stats_repository()`
- `routes_new.py` - добавлены новые endpoints
- `schemas.py` - добавлены новые Pydantic schemas

---

## Технические детали

- Полнотекстовый поиск использует PostgreSQL FTS с русским языковым профилем
- Экспорт поддерживает форматы CSV и JSON
- Все endpoints используют async/await
- Пагинация через limit/offset параметры
