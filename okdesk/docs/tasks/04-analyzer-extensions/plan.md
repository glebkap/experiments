# План реализации: Analyzer Service Extensions

**Task ID:** 04-analyzer-extensions
**Дата:** 25.11.2025
**Статус:** ✅ ЗАВЕРШЕНО

---

## Цель

Расширить Analyzer Service дополнительными endpoints для просмотра issues, статистики, полнотекстового поиска и экспорта.

---

## Пошаговый план

| Шаг | Описание | Статус |
|-----|----------|--------|
| 1 | Расширить IssueRepository interface | ✅ |
| 2 | Расширить ClusterRepository interface | ✅ |
| 3 | Создать StatsRepository interface | ✅ |
| 4 | Реализовать IssueRepositoryImpl | ✅ |
| 5 | Реализовать ClusterRepositoryImpl | ✅ |
| 6 | Реализовать StatsRepositoryImpl | ✅ |
| 7 | Добавить DTOs | ✅ |
| 8 | Добавить Pydantic Schemas | ✅ |
| 9 | Обновить Dependencies | ✅ |
| 10 | Добавить API Endpoints | ✅ |
| 11 | Проверка синтаксиса | ✅ |

---

## Критерии готовности

- [x] GET /analyzer/issues - список с фильтрами
- [x] GET /analyzer/issues/{id} - детали с messages
- [x] GET /analyzer/clusters/{id}/issues - issues в кластере
- [x] GET /analyzer/search/fulltext - FTS поиск
- [x] GET /analyzer/stats/processing - статистика обработки
- [x] GET /analyzer/stats/sources - по источникам
- [x] GET /analyzer/stats/clusters - по кластерам
- [x] GET /analyzer/stats/timeline - временная динамика
- [x] POST /analyzer/export - экспорт CSV/JSON

---

## Реализованные файлы

### Domain Layer
- `src/domain/repositories/issue_repository.py` - расширен новыми методами
- `src/domain/repositories/cluster_repository.py` - расширен новыми методами
- `src/domain/repositories/stats_repository.py` - **НОВЫЙ**
- `src/domain/repositories/__init__.py` - обновлен

### Infrastructure Layer
- `src/infrastructure/persistence/postgres/models.py` - добавлен SourceModel
- `src/infrastructure/persistence/postgres/issue_repository_impl.py` - расширен
- `src/infrastructure/persistence/postgres/cluster_repository_impl.py` - расширен
- `src/infrastructure/persistence/postgres/stats_repository_impl.py` - **НОВЫЙ**
- `src/infrastructure/dependencies.py` - добавлен get_stats_repository

### Application Layer (DTOs)
- `src/application/dto/issue_list_dto.py` - **НОВЫЙ**
- `src/application/dto/issue_detail_dto.py` - **НОВЫЙ**
- `src/application/dto/stats_dto.py` - **НОВЫЙ**
- `src/application/dto/export_dto.py` - **НОВЫЙ**
- `src/application/dto/__init__.py` - обновлен

### Interface Layer
- `src/interfaces/api/schemas.py` - добавлены новые Pydantic schemas
- `src/interfaces/api/routes_new.py` - добавлены новые endpoints
