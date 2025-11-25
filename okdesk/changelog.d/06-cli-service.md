# CLI Service - Changelog

**Task ID:** 06-cli-service
**Дата:** 25.11.2025
**Тип:** Feature - новый сервис

---

## Что реализовано

### 1. Инфраструктура проекта

- ✅ Создана структура проекта `services/cli/`
- ✅ Настроен `pyproject.toml` с зависимостями (Typer, Rich, httpx, Pydantic)
- ✅ Конфигурация через `.env` файл (Pydantic Settings)
- ✅ Entry point для установки: `support-cli`

### 2. API Client

Реализован полнофункциональный HTTP клиент для взаимодействия с backend API:

- **Базовая инфраструктура:**
  - `APIClient` с retry logic и error handling
  - Кастомные исключения: `APIError`, `ConnectionError`, `ValidationError`, `NotFoundError`, `ServerError`
  - Pydantic модели для всех API responses

- **Endpoints:**
  - Import: `import_okdesk()`, `import_telegram()`, `get_import_status()`, `list_imports()`
  - Pipeline: `process_pipeline()`, `get_pipeline_status()`
  - Clustering: `run_clustering()`, `get_clustering_info()`, `get_cluster_issues()`
  - Search: `search_similar()`, `search_fulltext()`, `list_issues()`, `get_issue()`
  - Stats: `get_stats_processing()`, `get_stats_clusters()`, `get_stats_sources()`, `get_stats_timeline()`
  - Export: `export_data()`

### 3. Форматирование вывода (Rich)

Реализованы модули для красивого вывода:

- **`styles.py`**: цветовая схема и стили (success, error, warning, info)
- **`formatters.py`**: форматирование дат, статусов, размеров файлов, длительности, процентов, similarity scores
- **`tables.py`**: Rich таблицы для imports, issues, search results, clusters, statistics, sources
- **`progress.py`**: прогресс-бары и спиннеры для длительных операций

### 4. Команды CLI (6 групп)

#### 4.1 Import Commands (`import`)
```bash
support-cli import okdesk <file> [--wait]
support-cli import telegram <file> [--wait]
support-cli import status <import-id>
support-cli import history [--limit]
```

#### 4.2 Pipeline Commands (`pipeline`)
```bash
support-cli pipeline process [--batch-size] [--device]
support-cli pipeline status
```

#### 4.3 Clustering Commands (`cluster`)
```bash
support-cli cluster run [--method] [--min-size] [--use-llm]
support-cli cluster list
support-cli cluster show <cluster-id> [--limit]
```

#### 4.4 Search Commands (`search`)
```bash
support-cli search similar <query> [--top-k] [--min-similarity]
support-cli search fulltext <query> [--limit] [--offset]
support-cli search list [--status] [--source] [--limit]
support-cli search show <issue-id> [--format] [--messages]
```

#### 4.5 Stats Commands (`stats`)
```bash
support-cli stats processing
support-cli stats clusters
support-cli stats sources
support-cli stats timeline [--from] [--to] [--granularity]
```

#### 4.6 Export Commands (`export`)
```bash
support-cli export data [--format] [--output] [--status] [--cluster] [--from] [--to]
```

### 5. Дополнительные команды

```bash
support-cli health      # Проверка доступности API
support-cli version     # Версия CLI
support-cli --help      # Справка по всем командам
```

### 6. Global Options

Все команды поддерживают:
- `--api-url` - переопределение API Gateway URL
- `--verbose` - подробный вывод
- `--quiet` - тихий режим

### 7. Docker и развертывание

- ✅ Multi-stage Dockerfile для минимального образа
- ✅ Поддержка volume mount для данных
- ✅ Entry point настроен на `support-cli`

### 8. Документация

- ✅ Полный README.md с примерами использования всех команд
- ✅ Документация конфигурации
- ✅ Troubleshooting секция
- ✅ Docstrings для всех команд (используются Typer для --help)

### 9. Тестирование

- ✅ Pytest fixtures (conftest.py)
- ✅ Mock API client для unit тестов
- ✅ Готовая инфраструктура для тестов

---

## Технические детали

### Архитектура

```
services/cli/
├── src/support_cli/
│   ├── client/              # API клиент
│   │   ├── api_client.py    # HTTP client с retry
│   │   ├── models.py        # Pydantic модели
│   │   └── exceptions.py    # Кастомные исключения
│   ├── commands/            # Typer команды (6 модулей)
│   │   ├── import_cmd.py
│   │   ├── pipeline_cmd.py
│   │   ├── clustering_cmd.py
│   │   ├── search_cmd.py
│   │   ├── stats_cmd.py
│   │   └── export_cmd.py
│   ├── output/              # Rich форматирование
│   │   ├── styles.py
│   │   ├── formatters.py
│   │   ├── tables.py
│   │   └── progress.py
│   ├── utils/               # Утилиты
│   │   ├── date_utils.py
│   │   └── validation.py
│   ├── config.py            # Pydantic Settings
│   └── main.py              # Entry point
├── tests/                   # Pytest тесты
├── Dockerfile               # Multi-stage build
├── pyproject.toml           # uv конфигурация
├── .env.example             # Пример конфигурации
└── README.md                # Документация
```

### Зависимости

- **Typer[all] >= 0.12.0** - CLI framework
- **Rich >= 13.7.0** - форматирование вывода
- **httpx >= 0.27.0** - HTTP client
- **Pydantic >= 2.8.0** - валидация данных
- **pydantic-settings >= 2.3.0** - конфигурация
- **python-dotenv >= 1.0.0** - загрузка .env

### Ключевые возможности

1. **Красивый вывод**: таблицы, цвета, прогресс-бары через Rich
2. **Error handling**: понятные сообщения об ошибках, exit codes
3. **Retry logic**: автоматические повторы при сбоях сети
4. **Health checks**: проверка доступности API перед командами
5. **Type hints**: полная типизация кода
6. **Docstrings**: документация для всех функций
7. **Context manager**: правильное управление ресурсами HTTP клиента

---

## Примеры использования

### Полный цикл обработки данных

```bash
# 1. Импорт
support-cli import okdesk /data/okdesk/out.jsonl --wait

# 2. Pipeline обработка
support-cli pipeline process --batch-size 100

# 3. Кластеризация
support-cli cluster run --method hdbscan --use-llm

# 4. Просмотр результатов
support-cli cluster list
support-cli stats processing
```

### Анализ проблемы

```bash
# Семантический поиск
support-cli search similar "проблема с оплатой" --top-k 10

# Детали issue
support-cli search show <issue-id>

# Кластер с похожими
support-cli cluster show <cluster-id>

# Экспорт
support-cli export data --format csv --cluster <cluster-id>
```

---

## Статус

✅ **Полностью реализовано** - все команды из спецификации готовы

### Что работает:

- ✅ Все 6 групп команд
- ✅ API Client для всех endpoints
- ✅ Rich форматирование
- ✅ Конфигурация через .env
- ✅ Docker support
- ✅ Документация
- ✅ Error handling
- ✅ Health checks

### Следующие шаги:

1. Интеграционные тесты с реальным API (после готовности API Gateway)
2. Unit тесты для команд (с моками)
3. CI/CD пайплайн для тестирования
4. Возможные улучшения:
   - Автодополнение в shell (через Typer)
   - Кэширование результатов
   - Интерактивный режим (prompt_toolkit)

---

## Зависимости

- ✅ Database Service (готов)
- ✅ Parser Service (готов)
- ✅ Analyzer Service (базовый готов)
- 🔄 API Gateway (в разработке) - CLI работает напрямую с сервисами или через Gateway

---

## Критерии готовности (выполнено)

- ✅ Все команды реализованы
- ✅ API Client поддерживает все endpoints
- ✅ Красивый вывод через Rich
- ✅ Конфигурация через .env
- ✅ Dockerfile готов
- ✅ README с примерами
- ✅ Type hints везде
- ✅ Docstrings для всех функций
- ✅ Error handling
- ✅ Health checks
- ✅ pytest fixtures

---

## Заметки

1. CLI готов к использованию как через `uv` локально, так и через Docker
2. Команды покрывают весь функционал системы (import → pipeline → clustering → search → stats → export)
3. Rich обеспечивает отличный UX с таблицами, цветами и прогресс-барами
4. API Client устойчив к сбоям (retry, timeout, error handling)
5. Документация подробная с множеством примеров
