# API Gateway Service - Декомпозиция задач

**Сервис:** support-api-gateway
**Приоритет:** Средний
**Технологии:** Python 3.12, FastAPI, httpx, uv

---

## Обзор

API Gateway - единая точка входа для всех клиентов. Маршрутизирует запросы к соответствующим сервисам.

---

## Структура

```
services/api-gateway/
├── src/
│   ├── routes/
│   │   ├── import_routes.py    # Proxy to Parser
│   │   ├── analyze_routes.py   # Proxy to Analyzer
│   │   └── query_routes.py     # Proxy to Query
│   ├── middleware/
│   │   ├── logging_middleware.py
│   │   └── cors_middleware.py
│   ├── health/
│   │   └── health_check.py
│   ├── config.py
│   └── main.py
├── tests/
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## Задачи

### 1. Настройка проекта

- [ ] Создать структуру
- [ ] Инициализировать uv проект
- [ ] Добавить зависимости: fastapi, uvicorn, httpx
- [ ] Создать Dockerfile

### 2. Конфигурация

- [ ] Создать config.py
  - URLs сервисов (Parser, Analyzer, Query)
  - Timeouts
  - CORS settings

### 3. Proxy Routes

- [ ] import_routes.py
  - Проксировать все /api/v1/import/* к Parser Service
  - Использовать httpx для HTTP запросов
  - Обработка ошибок

- [ ] analyze_routes.py
  - Проксировать все /api/v1/analyze/* к Analyzer Service

- [ ] query_routes.py
  - Проксировать все /api/v1/search, /api/v1/issues/*, /api/v1/stats/*, /api/v1/clusters/* к Query Service

### 4. Middleware

- [ ] Logging Middleware
  - Логировать все входящие запросы
  - Request ID для корреляции

- [ ] CORS Middleware
  - Разрешить запросы от GUI
  - Настроить allowed origins

### 5. Health Checks

- [ ] health_check.py
  - GET /api/v1/health
  - Проверить доступность Parser, Analyzer, Query
  - Вернуть статус каждого сервиса

### 6. Main Application

- [ ] main.py
  - FastAPI app
  - Include all routers
  - Add middleware
  - Exception handlers

### 7. Тестирование

- [ ] Unit тесты для routes (с моками)
- [ ] Интеграционные тесты с реальными сервисами

### 8. Документация

- [ ] README.md

---

## Критерии готовности

- [ ] Все routes работают
- [ ] Проксирование корректное
- [ ] Health checks работают
- [ ] Logging настроен
- [ ] CORS работает

---

## Зависимости

**Требует:**
- Parser Service готов
- Analyzer Service готов
- Query Service готов
