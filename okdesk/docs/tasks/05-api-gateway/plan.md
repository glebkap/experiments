# План реализации API Gateway Service

**Task ID:** 05-api-gateway
**Дата:** 25.11.2025
**Приоритет:** Средний
**Зависимости:** Parser Service (✅), Analyzer Service (✅)

---

## Цель

Реализовать простой API Gateway как единую точку входа для всех клиентов (GUI, CLI). Gateway выполняет роль **reverse proxy** и не содержит бизнес-логики.

---

## Принципы реализации

- ✅ Максимальная простота
- ✅ Только роутинг к сервисам Parser и Analyzer
- ✅ Никакой бизнес-логики
- ✅ Агрегация health checks
- ✅ CORS middleware для GUI
- ✅ Логирование всех proxy запросов

---

## Этапы реализации

### Этап 1: Подготовка структуры проекта ⏳

**Действия:**
1. Создать директорию `services/api-gateway/`
2. Инициализировать проект с `uv` и `pyproject.toml`
3. Создать структуру:
   ```
   services/api-gateway/
   ├── src/
   │   ├── __init__.py
   │   ├── config.py        # Конфигурация через pydantic-settings
   │   ├── proxy.py         # Reverse proxy логика
   │   ├── health.py        # Health check aggregation
   │   └── main.py          # FastAPI app
   ├── tests/
   │   ├── __init__.py
   │   ├── test_proxy.py    # Unit тесты для proxy
   │   └── test_health.py   # Unit тесты для health checks
   ├── Dockerfile
   ├── .dockerignore
   ├── pyproject.toml
   └── README.md
   ```

**Результат:** Базовая структура проекта готова

---

### Этап 2: Реализация конфигурации ⏳

**Файл:** `src/config.py`

**Функционал:**
- Использование `pydantic-settings` для управления конфигурацией
- URL сервисов: Parser (http://parser:8001), Analyzer (http://analyzer:8002)
- Настройки API Gateway: host, port
- CORS origins для GUI (http://localhost:3000)
- Request timeout (30 секунд по умолчанию)
- Загрузка из `.env` файла

**Результат:** Централизованная конфигурация через environment variables

---

### Этап 3: Реализация reverse proxy ⏳

**Файл:** `src/proxy.py`

**Функционал:**
- Функция `proxy_request()` для проксирования HTTP запросов
- Использование `httpx.AsyncClient` для асинхронных запросов
- Построение полного URL с query параметрами
- Передача headers (кроме host)
- Передача body для POST/PUT запросов
- Обработка ошибок:
  - `httpx.TimeoutException` → 504 Gateway Timeout
  - `httpx.RequestError` → 503 Service Unavailable
- Логирование всех proxy запросов

**Результат:** Универсальная функция для проксирования к любому сервису

---

### Этап 4: Реализация health checks ⏳

**Файл:** `src/health.py`

**Функционал:**
- Функция `check_service_health()` - проверка одного сервиса:
  - GET запрос к `{service_url}/health`
  - Измерение latency в миллисекундах
  - Возврат статуса: "ok" | "error"
  - Timeout 5 секунд
- Функция `aggregate_health_checks()` - агрегация для всех сервисов:
  - Параллельные запросы к Parser и Analyzer
  - Определение общего статуса:
    - "ok" - все сервисы работают
    - "degraded" - хотя бы один работает
    - "error" - все недоступны
  - Возврат детальной информации по каждому сервису

**Результат:** Мониторинг состояния всех микросервисов

---

### Этап 5: Реализация FastAPI приложения ⏳

**Файл:** `src/main.py`

**Функционал:**

**1. Инициализация:**
- FastAPI app с title, version, description
- CORS middleware с настройками из config
- Настройка логирования

**2. Health endpoints:**
- `GET /health` - health check самого Gateway
- `GET /api/v1/health` - агрегированный health check всех сервисов
- `GET /api/v1/health/parser` - health check Parser Service
- `GET /api/v1/health/analyzer` - health check Analyzer Service

**3. Proxy routes:**
- `ALL /api/v1/import/{path:path}` → Parser Service
  - Поддержка GET, POST, PUT, DELETE
- `ALL /api/v1/analyzer/{path:path}` → Analyzer Service
  - Поддержка GET, POST, PUT, DELETE

**4. Дополнительные endpoints:**
- `GET /` - информация о Gateway и доступных endpoints

**5. Error handling:**
- Global exception handler для логирования и возврата 500

**Результат:** Полнофункциональный API Gateway

---

### Этап 6: Настройка зависимостей ⏳

**Файл:** `pyproject.toml`

**Зависимости:**
- `fastapi>=0.100.0` - web framework
- `uvicorn[standard]>=0.23.0` - ASGI server
- `httpx>=0.24.0` - HTTP client для proxy
- `pydantic>=2.0.0` - data validation
- `pydantic-settings>=2.0.0` - settings management

**Dev зависимости:**
- `pytest>=7.4.0` - testing framework
- `pytest-asyncio>=0.21.0` - async test support
- `httpx>=0.24.0` - для тестовых запросов

**Результат:** Все необходимые зависимости определены

---

### Этап 7: Написание unit тестов ⏳

**Файл:** `tests/test_proxy.py`

**Тесты:**
- `test_health()` - проверка `/health` endpoint
- `test_root()` - проверка `/` endpoint
- `test_proxy_to_parser()` - мок проксирования к Parser
- `test_proxy_to_analyzer()` - мок проксирования к Analyzer
- `test_proxy_timeout()` - обработка timeout
- `test_proxy_service_unavailable()` - обработка недоступности сервиса

**Файл:** `tests/test_health.py`

**Тесты:**
- `test_check_service_health_ok()` - успешный health check
- `test_check_service_health_error()` - сервис недоступен
- `test_aggregate_health_all_ok()` - все сервисы работают
- `test_aggregate_health_degraded()` - один сервис недоступен
- `test_aggregate_health_all_error()` - все сервисы недоступны

**Результат:** Покрытие тестами >80%

---

### Этап 8: Docker контейнеризация ⏳

**Файл:** `Dockerfile`

**Содержимое:**
- Base image: `python:3.12-slim`
- Установка `uv`
- Копирование `pyproject.toml` и `src/`
- Установка зависимостей через `uv pip install --system -e .`
- Expose port 8000
- CMD: `uvicorn src.main:app --host 0.0.0.0 --port 8000`

**Файл:** `.dockerignore`
- Исключение ненужных файлов из образа

**Результат:** Docker образ API Gateway готов к сборке

---

### Этап 9: Интеграция с Docker Compose ⏳

**Действия:**
1. Проверить наличие главного `docker-compose.yml` в корне
2. Добавить сервис `api-gateway`:
   ```yaml
   api-gateway:
     build: ./services/api-gateway
     container_name: support-api-gateway
     depends_on:
       - parser
       - analyzer
     environment:
       PARSER_URL: http://parser:8001
       ANALYZER_URL: http://analyzer:8002
     ports:
       - "8000:8000"
     networks:
       - support-network
   ```

**Результат:** API Gateway интегрирован в общую инфраструктуру

---

### Этап 10: Тестирование и проверка ⏳

**Действия:**
1. Проверить статус Parser и Analyzer сервисов
2. Собрать Docker образ API Gateway
3. Запустить API Gateway через docker-compose
4. Проверить endpoints:
   - `GET http://localhost:8000/health`
   - `GET http://localhost:8000/api/v1/health`
   - `GET http://localhost:8000/api/v1/health/parser`
   - `GET http://localhost:8000/api/v1/health/analyzer`
5. Проверить proxy:
   - Запрос к Parser через Gateway
   - Запрос к Analyzer через Gateway
6. Проверить логи API Gateway
7. Запустить unit тесты: `pytest tests/`

**Результат:** API Gateway работает корректно, все тесты проходят

---

### Этап 11: Документация ⏳

**Файл:** `services/api-gateway/README.md`

**Содержимое:**
- Описание сервиса
- Архитектура и принципы
- Структура проекта
- Конфигурация через environment variables
- API endpoints с примерами
- Запуск локально и в Docker
- Тестирование
- Troubleshooting

**Результат:** Полная документация для разработчиков

---

## Критерии готовности

- [⏳] Структура проекта создана
- [⏳] Конфигурация через pydantic-settings реализована
- [⏳] Reverse proxy функционал реализован
- [⏳] Health checks реализованы
- [⏳] FastAPI приложение реализовано
- [⏳] Зависимости определены в pyproject.toml
- [⏳] Unit тесты написаны и проходят
- [⏳] Dockerfile создан
- [⏳] Интеграция с docker-compose выполнена
- [⏳] API Gateway запускается и работает
- [⏳] Proxy к Parser работает
- [⏳] Proxy к Analyzer работает
- [⏳] CORS настроен корректно
- [⏳] Логирование работает
- [⏳] Документация написана

---

## Оценка времени

| Этап | Время |
|------|-------|
| Структура проекта | 15 мин |
| Конфигурация | 15 мин |
| Reverse proxy | 45 мин |
| Health checks | 30 мин |
| FastAPI app | 1 час |
| Зависимости | 10 мин |
| Unit тесты | 1.5 часа |
| Docker | 30 мин |
| Docker Compose | 15 мин |
| Тестирование | 1 час |
| Документация | 30 мин |

**Итого: ~6-7 часов чистого времени**

---

## Риски и ограничения

1. **Зависимость от других сервисов:**
   - Parser и Analyzer должны быть запущены и доступны
   - Mitigation: health checks покажут проблемы

2. **Timeout настройки:**
   - Некоторые операции Analyzer могут быть долгими (кластеризация)
   - Mitigation: настраиваемый timeout через config

3. **CORS для GUI:**
   - Нужно добавить правильные origins
   - Mitigation: настраиваемый список через environment

4. **Логирование:**
   - Может быть много логов при активном использовании
   - Mitigation: настройка уровня логирования через environment

---

## Следующие шаги после реализации

1. Реализация CLI Service (использует API Gateway)
2. Реализация GUI Service (использует API Gateway)
3. Добавление rate limiting (опционально)
4. Добавление authentication (опционально)
5. Мониторинг и метрики (Prometheus, опционально)

---

## Примечания

- API Gateway максимально простой - это его преимущество
- Никакой бизнес-логики - только routing и monitoring
- Все сложные операции остаются в Parser и Analyzer
- Health checks позволяют быстро диагностировать проблемы в системе
