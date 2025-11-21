# План реализации Parser Service

**Task ID:** 02-parser-service
**Дата создания:** 21.11.2025
**Приоритет:** Высокий (блокирует Analyzer Service)
**Статус:** Планирование

---

## Цель

Реализовать Parser Service - микросервис для импорта данных из OKDesk (JSONL) и Telegram (JSON) с использованием архитектуры DDD (Domain-Driven Design).

---

## Предварительные условия

- [x] Database Service запущен и работает
- [x] Схема БД создана (migrations выполнены)
- [x] Документация изучена (PRD, ARCHITECTURE, 02-parser-service.md)

---

## Этапы выполнения

### Этап 1: Инициализация проекта ⏳

**Задачи:**
- [ ] 1.1 Создать структуру директорий согласно DDD
- [ ] 1.2 Инициализировать uv проект
- [ ] 1.3 Добавить зависимости через uv
- [ ] 1.4 Создать `pyproject.toml` с настройками
- [ ] 1.5 Создать `.env.example` для конфигурации
- [ ] 1.6 Создать базовый `config.py`

**Критерии готовности:**
- Структура директорий соответствует DDD
- uv проект инициализирован
- Все зависимости установлены
- Конфигурация настроена

**Время:** ~30 минут

---

### Этап 2: Domain Layer (Бизнес-логика) ⏳

**Задачи:**
- [ ] 2.1 Создать domain models (entities):
  - [ ] `source.py` - SourceType enum, Source dataclass
  - [ ] `issue.py` - IssueStatus enum, Issue dataclass
  - [ ] `message.py` - AuthorType enum, Message dataclass
  - [ ] `import_job.py` - ImportStatus enum, ImportJob dataclass
- [ ] 2.2 Создать repository interfaces:
  - [ ] `source_repository.py` - Abstract base class
  - [ ] `issue_repository.py` - Abstract base class
  - [ ] `message_repository.py` - Abstract base class
  - [ ] `import_repository.py` - Abstract base class
- [ ] 2.3 Создать domain services:
  - [ ] `deduplication_service.py` - Проверка дубликатов
  - [ ] `import_service.py` - Координация импорта

**Критерии готовности:**
- Все domain models созданы с типами данных
- Repository interfaces определены
- Domain services содержат бизнес-логику без зависимостей от инфраструктуры

**Время:** ~2 часа

---

### Этап 3: Infrastructure Layer (Persistence) ⏳

**Задачи:**
- [ ] 3.1 Создать `database.py`:
  - [ ] Настройка SQLAlchemy engine
  - [ ] SessionLocal factory
  - [ ] Dependency для FastAPI
- [ ] 3.2 Создать SQLAlchemy `models.py`:
  - [ ] SourceModel (маппинг на таблицу sources)
  - [ ] IssueModel (маппинг на таблицу issues)
  - [ ] MessageModel (маппинг на таблицу messages)
  - [ ] ImportModel (маппинг на таблицу imports)
- [ ] 3.3 Реализовать repository implementations:
  - [ ] `source_repository_impl.py`
  - [ ] `issue_repository_impl.py`
  - [ ] `message_repository_impl.py`
  - [ ] `import_repository_impl.py`

**Критерии готовности:**
- SQLAlchemy модели соответствуют схеме БД
- Все repositories реализованы
- Маппинг domain models ↔ DB models работает

**Время:** ~3 часа

---

### Этап 4: Infrastructure Layer (Parsers) ⏳

**Задачи:**
- [ ] 4.1 Создать `okdesk_parser.py`:
  - [ ] Метод `parse_file(file_path)` - построчное чтение JSONL
  - [ ] Метод `extract_issue(data)` - извлечение issue
  - [ ] Метод `extract_comments(data)` - извлечение комментариев
  - [ ] Очистка HTML через BeautifulSoup
- [ ] 4.2 Создать `telegram_parser.py`:
  - [ ] Метод `parse_file(file_path)` - чтение JSON
  - [ ] Метод `extract_messages(data)` - извлечение сообщений
  - [ ] Маппинг структуры Telegram на domain models

**Критерии готовности:**
- OKDesk parser корректно читает JSONL
- Telegram parser корректно читает JSON
- HTML очищается от тегов

**Время:** ~2 часа

---

### Этап 5: Infrastructure Layer (HTTP Client) ⏳

**Задачи:**
- [ ] 5.1 Создать `analyzer_client.py`:
  - [ ] Метод `analyze_batch(message_ids, batch_size)`
  - [ ] POST запрос к Analyzer Service
  - [ ] Обработка ошибок и retry логика
  - [ ] Использование httpx для async запросов

**Критерии готовности:**
- HTTP client может отправлять запросы к Analyzer Service
- Ошибки обрабатываются корректно
- Retry логика работает

**Время:** ~1 час

---

### Этап 6: Application Layer (Use Cases) ⏳

**Задачи:**
- [ ] 6.1 Создать DTOs в `import_dto.py`:
  - [ ] ImportRequest
  - [ ] ImportResponse
  - [ ] ImportStats
  - [ ] ImportProgress
- [ ] 6.2 Создать `import_okdesk.py`:
  - [ ] Класс ImportOKDeskUseCase
  - [ ] Dependency Injection через конструктор
  - [ ] Метод `execute(file_path, source_id)`
  - [ ] Алгоритм импорта с транзакциями
- [ ] 6.3 Создать `import_telegram.py`:
  - [ ] Класс ImportTelegramUseCase
  - [ ] Аналогичная логика
- [ ] 6.4 Создать `get_import_status.py`:
  - [ ] Класс GetImportStatusUseCase
- [ ] 6.5 Создать `list_imports.py`:
  - [ ] Класс ListImportsUseCase с пагинацией

**Критерии готовности:**
- Use cases содержат всю бизнес-логику
- Зависимости инжектятся через конструктор
- Транзакции управляются корректно

**Время:** ~3 часа

---

### Этап 7: Interface Layer (API) ⏳

**Задачи:**
- [ ] 7.1 Создать Pydantic schemas в `schemas.py`:
  - [ ] ImportResponse
  - [ ] ImportStatusResponse
  - [ ] ImportListResponse
  - [ ] ErrorResponse
- [ ] 7.2 Создать FastAPI routes в `routes.py`:
  - [ ] POST /api/v1/import/okdesk
  - [ ] POST /api/v1/import/telegram
  - [ ] GET /api/v1/import/{id}
  - [ ] GET /api/v1/imports
- [ ] 7.3 Настроить обработку multipart/form-data
- [ ] 7.4 Добавить валидацию и обработку ошибок

**Критерии готовности:**
- Все API endpoints работают
- Валидация входных данных настроена
- Ошибки возвращают правильные HTTP коды

**Время:** ~2 часа

---

### Этап 8: Dependency Injection и Main ⏳

**Задачи:**
- [ ] 8.1 Создать `dependencies.py`:
  - [ ] Фабрики для всех repositories
  - [ ] Фабрики для parsers
  - [ ] Фабрики для use cases
  - [ ] get_db() для session management
- [ ] 8.2 Создать `main.py`:
  - [ ] FastAPI app initialization
  - [ ] Include routers
  - [ ] Health check endpoint
  - [ ] CORS middleware
  - [ ] Exception handlers
  - [ ] Logging setup

**Критерии готовности:**
- DI настроен через FastAPI Depends
- Приложение запускается
- Health check отвечает

**Время:** ~1.5 часа

---

### Этап 9: Docker и контейнеризация ⏳

**Задачи:**
- [ ] 9.1 Создать Dockerfile:
  - [ ] Multi-stage build
  - [ ] Использование uv для установки зависимостей
  - [ ] Оптимизация размера образа
- [ ] 9.2 Обновить docker-compose.yml:
  - [ ] Добавить parser service
  - [ ] Настроить зависимости (db)
  - [ ] Пробросить volumes для data/
  - [ ] Настроить environment variables
- [ ] 9.3 Создать .dockerignore

**Критерии готовности:**
- Docker образ собирается
- Сервис запускается в Docker Compose
- Доступен на порту 8001

**Время:** ~1 час

---

### Этап 10: Тестирование ⏳

**Задачи:**
- [ ] 10.1 Создать unit тесты:
  - [ ] `test_deduplication_service.py`
  - [ ] `test_import_okdesk_use_case.py` (с моками)
  - [ ] `test_okdesk_parser.py`
- [ ] 10.2 Создать integration тесты:
  - [ ] `test_import_okdesk.py` (с тестовой БД)
  - [ ] `test_api_endpoints.py`
- [ ] 10.3 Создать test fixtures:
  - [ ] `okdesk_sample.jsonl` (3-5 обращений)
  - [ ] `telegram_sample.json`
- [ ] 10.4 Настроить pytest.ini
- [ ] 10.5 Запустить все тесты и проверить coverage

**Критерии готовности:**
- Unit тесты покрывают domain и application layers
- Integration тесты проверяют работу с БД
- Coverage > 80%

**Время:** ~3 часа

---

### Этап 11: Документация ⏳

**Задачи:**
- [ ] 11.1 Создать `services/parser/README.md`:
  - [ ] Описание сервиса
  - [ ] DDD архитектура (диаграмма слоев)
  - [ ] Установка и запуск
  - [ ] API документация
  - [ ] Примеры использования curl
- [ ] 11.2 Добавить docstrings:
  - [ ] Ко всем классам
  - [ ] Ко всем публичным методам
  - [ ] К API endpoints (для Swagger)
- [ ] 11.3 Создать changelog в `changelog.d/02-parser-service.md`

**Критерии готовности:**
- README полный и понятный
- Docstrings присутствуют
- Swagger UI генерируется автоматически
- Changelog создан

**Время:** ~1.5 часа

---

### Этап 12: Интеграционное тестирование с БД ⏳

**Задачи:**
- [ ] 12.1 Запустить Database Service
- [ ] 12.2 Запустить Parser Service в Docker
- [ ] 12.3 Протестировать импорт реального файла OKDesk
- [ ] 12.4 Проверить данные в БД через psql
- [ ] 12.5 Протестировать дедупликацию (повторный импорт)
- [ ] 12.6 Проверить вызов Analyzer Service (мок или заглушка)

**Критерии готовности:**
- Данные корректно импортируются в БД
- Дедупликация работает
- Нет ошибок в логах

**Время:** ~1 час

---

## Общая оценка времени

**Всего:** ~21.5 часа чистого времени
**С учетом отладки и доработок:** ~25-30 часов
**Календарное время:** 3-5 дней

---

## Риски и зависимости

### Риски:
1. **Формат данных OKDesk может отличаться** от документации
   - Митигация: Использовать реальный файл для тестирования
2. **Проблемы с HTML парсингом** в сообщениях
   - Митигация: BeautifulSoup справляется с невалидным HTML
3. **Analyzer Service еще не реализован**
   - Митигация: Сделать вызов опциональным, использовать мок

### Зависимости:
- Database Service должен быть запущен
- Структура БД должна соответствовать schema
- Тестовые данные (okdesk sample files)

---

## Критерии приемки

- [ ] Все этапы плана выполнены
- [ ] Parser Service запускается в Docker
- [ ] API endpoints работают (проверены через curl/Postman)
- [ ] Импорт OKDesk файла работает корректно
- [ ] Импорт Telegram файла работает корректно
- [ ] Дедупликация предотвращает повторное добавление
- [ ] Unit тесты проходят (coverage > 80%)
- [ ] Integration тесты проходят
- [ ] Документация создана
- [ ] Changelog создан
- [ ] Код следует DDD принципам
- [ ] Готов коммит с префиксом [02-parser-service]

---

## Следующие шаги

После завершения Parser Service:
1. Реализовать Analyzer Service (pydantic_ai)
2. Интегрировать Parser → Analyzer через HTTP
3. Создать CLI команды для импорта

---

## Примечания

- Следовать DDD строго (не смешивать слои)
- Использовать type hints везде (Python 3.12)
- Логировать все важные операции
- Обрабатывать ошибки gracefully
- Писать тесты параллельно с кодом
