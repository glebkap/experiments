# API Gateway Service

**Единая точка входа для всех клиентов системы анализа намерений пользователей службы поддержки.**

## Описание

API Gateway - это простой reverse proxy сервис, который маршрутизирует запросы к соответствующим микросервисам (Parser и Analyzer). Gateway не содержит бизнес-логики и выполняет только функции роутинга и мониторинга.

## Принципы

- ✅ Максимальная простота
- ✅ Только роутинг к сервисам
- ✅ Никакой бизнес-логики
- ✅ Агрегация health checks
- ✅ CORS middleware для веб-клиентов
- ✅ Логирование всех proxy запросов

## Архитектура

```
┌──────────┐
│  Client  │ (GUI / CLI / curl)
└────┬─────┘
     │ HTTP :8000
     ▼
┌─────────────────┐
│  API Gateway    │ FastAPI
│  (port 8000)    │
└────┬────────────┘
     │
     ├─────────► Parser Service    (port 8001)
     └─────────► Analyzer Service  (port 8002)
```

## API Endpoints

### Health Checks

- `GET /health` - Health check самого Gateway
- `GET /api/v1/health` - Агрегированный health check всех сервисов
- `GET /api/v1/health/parser` - Health check Parser Service
- `GET /api/v1/health/analyzer` - Health check Analyzer Service

### Proxied Routes

#### Parser Service

Все запросы к `/api/v1/import/*` проксируются к Parser Service:

- `POST /api/v1/import/okdesk` - Загрузка данных OKDesk
- `POST /api/v1/import/telegram` - Загрузка данных Telegram
- `GET /api/v1/import/{id}` - Статус импорта
- `GET /api/v1/import` - История импортов

#### Analyzer Service

Все запросы к `/api/v1/analyzer/*` проксируются к Analyzer Service:

- `POST /api/v1/analyzer/pipeline/process` - Запуск pipeline обработки
- `GET /api/v1/analyzer/pipeline/status` - Статус pipeline
- `POST /api/v1/analyzer/clustering/run` - Запуск кластеризации
- `GET /api/v1/analyzer/clustering/info` - Информация о кластерах
- `POST /api/v1/analyzer/search/similar` - Семантический поиск
- `GET /api/v1/analyzer/issues` - Список обращений
- `GET /api/v1/analyzer/issues/{id}` - Детали обращения
- `GET /api/v1/analyzer/clusters/{id}/issues` - Issues в кластере
- `GET /api/v1/analyzer/search/fulltext` - Полнотекстовый поиск
- `GET /api/v1/analyzer/stats/*` - Статистика
- `POST /api/v1/analyzer/export` - Экспорт данных

### Documentation

- `GET /docs` - Swagger UI
- `GET /redoc` - ReDoc UI
- `GET /` - Gateway информация и список endpoints

## Конфигурация

API Gateway настраивается через environment variables или `.env` файл.

### Environment Variables

```env
# Service URLs
PARSER_URL=http://parser:8001
ANALYZER_URL=http://analyzer:8002

# Server Settings
HOST=0.0.0.0
PORT=8000

# CORS Settings
CORS_ORIGINS=["http://localhost:3000","http://localhost:8000"]

# Timeouts (seconds)
REQUEST_TIMEOUT=30
HEALTH_CHECK_TIMEOUT=5

# Logging
LOG_LEVEL=INFO
```

## Запуск

### С Docker Compose (рекомендуется)

```bash
# Из корневой директории проекта
docker compose up -d api-gateway

# Проверка логов
docker compose logs -f api-gateway

# Проверка health
curl http://localhost:8000/health
```

### Локальная разработка

```bash
cd services/api-gateway

# Установка зависимостей
uv sync

# Создание .env файла
cp .env.example .env

# Запуск сервера
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

## Тестирование

### Unit тесты

```bash
cd services/api-gateway

# Запуск всех тестов
uv run pytest

# Запуск с coverage
uv run pytest --cov=src --cov-report=html

# Запуск конкретного теста
uv run pytest tests/test_proxy.py::test_gateway_health -v
```

### Интеграционное тестирование

```bash
# Проверка Gateway
curl http://localhost:8000/health

# Проверка агрегированного health check
curl http://localhost:8000/api/v1/health

# Проверка proxy к Parser
curl http://localhost:8000/api/v1/import

# Проверка proxy к Analyzer
curl http://localhost:8000/api/v1/analyzer/pipeline/status
```

## Структура проекта

```
services/api-gateway/
├── src/
│   ├── __init__.py
│   ├── config.py        # Конфигурация через pydantic-settings
│   ├── proxy.py         # Reverse proxy логика
│   ├── health.py        # Health check aggregation
│   └── main.py          # FastAPI приложение
├── tests/
│   ├── __init__.py
│   ├── test_proxy.py    # Тесты proxy функциональности
│   └── test_health.py   # Тесты health checks
├── Dockerfile
├── .dockerignore
├── .env.example
├── pyproject.toml
└── README.md
```

## Обработка ошибок

Gateway возвращает следующие коды ошибок:

- **503 Service Unavailable** - Целевой сервис недоступен (connection error)
- **504 Gateway Timeout** - Целевой сервис не ответил вовремя (timeout)
- **500 Internal Server Error** - Внутренняя ошибка Gateway

Все ошибки логируются с полной информацией для отладки.

## Логирование

Gateway логирует:

- Все proxy запросы с методом, путем и целевым URL
- Ошибки подключения к сервисам
- Таймауты
- Результаты health checks
- Startup/shutdown события

Уровень логирования настраивается через `LOG_LEVEL` (DEBUG, INFO, WARNING, ERROR).

## Мониторинг

### Health Check Статусы

- **ok** - Все сервисы работают нормально
- **degraded** - Хотя бы один сервис работает (частичная доступность)
- **error** - Все сервисы недоступны

### Примеры ответов

**Все сервисы работают:**

```json
{
  "status": "ok",
  "services": {
    "parser": {
      "status": "ok",
      "latency_ms": 8.3,
      "details": {"service": "parser"}
    },
    "analyzer": {
      "status": "ok",
      "latency_ms": 12.5,
      "details": {"service": "analyzer"}
    }
  }
}
```

**Один сервис недоступен (degraded):**

```json
{
  "status": "degraded",
  "services": {
    "parser": {
      "status": "ok",
      "latency_ms": 9.1,
      "details": {"service": "parser"}
    },
    "analyzer": {
      "status": "error",
      "latency_ms": 5002.3,
      "details": {"error": "timeout"}
    }
  }
}
```

## Troubleshooting

### Gateway не запускается

1. Проверьте, что порт 8000 свободен:
   ```bash
   netstat -tuln | grep 8000
   ```

2. Проверьте логи:
   ```bash
   docker compose logs api-gateway
   ```

### Parser или Analyzer недоступны

1. Проверьте health check:
   ```bash
   curl http://localhost:8000/api/v1/health
   ```

2. Проверьте, что сервисы запущены:
   ```bash
   docker compose ps parser analyzer
   ```

3. Проверьте network connectivity:
   ```bash
   docker compose exec api-gateway ping parser
   docker compose exec api-gateway ping analyzer
   ```

### Timeout errors

1. Увеличьте `REQUEST_TIMEOUT` в .env
2. Проверьте производительность целевых сервисов
3. Проверьте сетевую задержку между контейнерами

## Производительность

- **Latency overhead**: ~1-5ms (reverse proxy overhead)
- **Throughput**: Зависит от целевых сервисов
- **Concurrent connections**: Неограниченно (asyncio)

## Безопасность

- CORS настроен для разрешенных origins
- Нет аутентификации (предполагается внутренняя сеть)
- Логирование всех запросов для аудита
- Изоляция в Docker network

## Future Improvements

Потенциальные улучшения (не реализованы в MVP):

- Rate limiting
- Authentication/Authorization
- Request/Response caching
- Metrics collection (Prometheus)
- Distributed tracing (OpenTelemetry)
- Circuit breaker pattern
- Load balancing (если несколько инстансов)

## Лицензия

Часть проекта "Support System".
