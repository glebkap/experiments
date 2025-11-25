# ⚠️ Query Service - УСТАРЕЛ ⚠️

**Статус:** ❌ УПРАЗДНЕН
**Дата упразднения:** 25.11.2025
**Причина:** Функциональность перенесена в Analyzer Service

---

## ⚠️ ВАЖНО

Этот документ оставлен для исторической справки. **НЕ ИСПОЛЬЗУЙТЕ** его для разработки.

**Вместо Query Service используйте:**

- **[04-analyzer-extensions.md](04-analyzer-extensions.md)** - расширение Analyzer Service
- **[05-api-gateway.md](05-api-gateway.md)** - API Gateway для роутинга

---

## Архитектурное решение

После анализа было принято решение **НЕ создавать отдельный Query Service** по следующим причинам:

1. **Избыточная сложность** - Query Service не содержит бизнес-логики, только read-only запросы к БД
2. **Дублирование кода** - репозитории и модели уже есть в Analyzer
3. **Упрощение архитектуры** - меньше сервисов = проще поддержка
4. **ChromaDB уже в Analyzer** - семантический поиск не нужно дублировать

### Новая архитектура

```
┌─────────────┐      ┌──────────────────────┐
│   Parser    │      │   Analyzer Service   │
│   Service   │      │  ┌─────────────────┐ │
│             │      │  │ Pipeline + ML   │ │
│             │      │  │ ChromaDB Search │ │
└──────┬──────┘      │  │ Issues API      │ │
       │             │  │ Stats API       │ │
       │             │  │ Export API      │ │
       └─────────────┴──┴─────────────────┘─┘
                      │
                      ▼
                 PostgreSQL + ChromaDB
```

Все функции Query Service теперь реализованы как расширения Analyzer Service.

---

## Миграция функций

| Функция Query Service | Новое место | Endpoint |
|----------------------|-------------|----------|
| Просмотр issues | Analyzer Extensions | GET /analyzer/issues |
| Детали issue | Analyzer Extensions | GET /analyzer/issues/{id} |
| Семантический поиск | Analyzer (готов) | POST /analyzer/search/similar |
| Полнотекстовый поиск | Analyzer Extensions | GET /analyzer/search/fulltext |
| Просмотр кластеров | Analyzer Extensions | GET /analyzer/clusters/{id}/issues |
| Статистика обработки | Analyzer Extensions | GET /analyzer/stats/* |
| Экспорт данных | Analyzer Extensions | POST /analyzer/export |

---

## Исходный план (для справки)

<details>
<summary>Развернуть исходный план Query Service</summary>

### Оригинальные требования

Query Service предназначался для:

- Поиск и фильтрация issues
- Статистика по различным параметрам
- Экспорт данных в CSV/JSON
- Read-only операции с БД

### Почему не реализован

Все эти функции логичнее расположить в Analyzer Service, так как:

1. Analyzer уже работает с теми же таблицами
2. ChromaDB для семантического поиска находится в Analyzer
3. Нет смысла дублировать репозитории
4. API Gateway может роутить запросы напрямую к Analyzer

</details>
