# План разработки CLI Service

**Task ID:** 06-cli-service
**Приоритет:** Средний
**Статус:** В разработке
**Дата создания:** 25.11.2025

---

## 1. Обзор задачи

Разработать полнофункциональный CLI интерфейс для взаимодействия с системой анализа обращений. CLI должен предоставлять удобный доступ ко всем операциям: импорт данных, запуск pipeline обработки, кластеризация, поиск, просмотр данных, статистика и экспорт.

### Технологический стек
- **Python 3.12**
- **Typer** - современный CLI framework с поддержкой типов
- **Rich** - красивое форматирование вывода (таблицы, прогресс-бары, цвета)
- **httpx** - HTTP клиент для взаимодействия с API
- **uv** - управление зависимостями
- **pydantic** - валидация конфигурации

### Способы запуска
1. **Docker** - через docker-compose (для production)
2. **uv** - локальная установка для разработки (`uv run support-cli`)

---

## 2. Архитектура CLI Service

### 2.1 Структура проекта

```
services/cli/
├── .env.example              # Пример конфигурации
├── .gitignore
├── Dockerfile                # Multi-stage build
├── pyproject.toml            # uv конфигурация
├── README.md                 # Документация
├── src/
│   └── support_cli/
│       ├── __init__.py
│       ├── main.py           # Entry point, Typer app
│       ├── config.py         # Конфигурация из .env
│       ├── client/
│       │   ├── __init__.py
│       │   ├── api_client.py         # HTTP client для API Gateway
│       │   ├── exceptions.py         # Кастомные исключения
│       │   └── models.py             # Pydantic модели для API responses
│       ├── commands/
│       │   ├── __init__.py
│       │   ├── import_cmd.py         # Команды импорта
│       │   ├── pipeline_cmd.py       # Pipeline обработка
│       │   ├── clustering_cmd.py     # Кластеризация
│       │   ├── search_cmd.py         # Поиск и просмотр
│       │   ├── stats_cmd.py          # Статистика
│       │   └── export_cmd.py         # Экспорт данных
│       ├── output/
│       │   ├── __init__.py
│       │   ├── formatters.py         # Форматирование данных
│       │   ├── tables.py             # Rich таблицы
│       │   ├── progress.py           # Прогресс-бары
│       │   └── styles.py             # Цветовая схема
│       └── utils/
│           ├── __init__.py
│           ├── date_utils.py         # Работа с датами
│           └── validation.py         # Валидация входных данных
└── tests/
    ├── __init__.py
    ├── conftest.py               # Pytest fixtures
    ├── test_api_client.py        # Тесты API client
    ├── test_commands/
    │   ├── test_import.py
    │   ├── test_pipeline.py
    │   ├── test_search.py
    │   └── test_stats.py
    └── test_formatters.py
```

### 2.2 Принципы разработки

1. **Разделение ответственности:**
   - `client/` - взаимодействие с API
   - `commands/` - бизнес-логика команд
   - `output/` - форматирование и отображение

2. **Dependency Injection:**
   - APIClient передается через Typer context
   - Легко мокировать в тестах

3. **Обработка ошибок:**
   - Кастомные исключения с понятными сообщениями
   - Graceful degradation при недоступности API
   - Подробный режим с `--verbose`

4. **User Experience:**
   - Прогресс-бары для длительных операций
   - Цветной вывод для статусов (успех=green, ошибка=red)
   - Таблицы Rich для структурированных данных
   - Интерактивные подтверждения для опасных операций

---

## 3. Детальный план разработки

### Этап 1: Инфраструктура и настройка проекта

#### 1.1 Инициализация проекта
- [ ] Создать структуру директорий `services/cli/`
- [ ] Инициализировать uv проект: `uv init`
- [ ] Настроить `pyproject.toml`:
  - Зависимости: typer, rich, httpx, pydantic, python-dotenv
  - Entry point: `support-cli = "support_cli.main:app"`
  - Dev зависимости: pytest, pytest-mock, pytest-asyncio
- [ ] Создать `.env.example` с параметрами:
  ```
  API_GATEWAY_URL=http://localhost:8000
  DEFAULT_BATCH_SIZE=100
  REQUEST_TIMEOUT=300
  LOG_LEVEL=INFO
  ```
- [ ] Создать `.gitignore`
- [ ] Создать `Dockerfile` (multi-stage: build + runtime)

#### 1.2 Конфигурация
- [ ] `config.py`:
  - Класс `Settings` с Pydantic BaseSettings
  - Загрузка из `.env` файла
  - Валидация обязательных параметров
  - Singleton паттерн для глобального доступа

### Этап 2: API Client

#### 2.1 Базовая инфраструктура
- [ ] `client/exceptions.py`:
  - `APIError` - базовое исключение
  - `ConnectionError` - ошибка подключения
  - `ValidationError` - ошибка валидации
  - `NotFoundError` - ресурс не найден
  - `ServerError` - ошибка сервера

- [ ] `client/models.py`:
  - Pydantic модели для всех API responses:
    - `ImportResponse`, `ImportStatus`, `ImportHistory`
    - `PipelineStatus`, `PipelineResponse`
    - `ClusteringInfo`, `ClusterDetails`
    - `SearchResult`, `IssueDetails`, `IssueListItem`
    - `StatsResponse` (processing, clusters, sources, timeline)
    - `ExportResponse`

#### 2.2 HTTP Client
- [ ] `client/api_client.py`:
  - Класс `APIClient` с httpx.Client
  - Методы для всех endpoints:

    **Import endpoints:**
    - `import_okdesk(file_path: Path) -> ImportResponse`
    - `import_telegram(file_path: Path) -> ImportResponse`
    - `get_import_status(import_id: str) -> ImportStatus`
    - `list_imports(limit: int = 50) -> List[ImportHistory]`

    **Pipeline endpoints:**
    - `process_pipeline(batch_size: int, device: str) -> PipelineResponse`
    - `get_pipeline_status() -> PipelineStatus`

    **Clustering endpoints:**
    - `run_clustering(method: str, min_cluster_size: int, use_llm: bool) -> ClusteringInfo`
    - `get_clustering_info() -> ClusteringInfo`
    - `get_cluster_issues(cluster_id: str, limit: int) -> List[IssueListItem]`

    **Search endpoints:**
    - `search_similar(query: str, top_k: int, min_similarity: float) -> List[SearchResult]`
    - `search_fulltext(query: str, limit: int, offset: int) -> List[IssueListItem]`
    - `list_issues(status: str, source_id: str, limit: int, offset: int) -> List[IssueListItem]`
    - `get_issue(issue_id: str) -> IssueDetails`

    **Stats endpoints:**
    - `get_stats_processing() -> StatsResponse`
    - `get_stats_clusters() -> StatsResponse`
    - `get_stats_sources() -> StatsResponse`
    - `get_stats_timeline(from_date: date, to_date: date, granularity: str) -> StatsResponse`

    **Export endpoints:**
    - `export_data(format: str, filters: dict) -> ExportResponse`

  - Общие методы:
    - `_request(method, endpoint, **kwargs)` - базовый метод с обработкой ошибок
    - `_handle_error(response)` - парсинг ошибок API
    - `health_check() -> bool` - проверка доступности API

### Этап 3: Команды импорта

#### 3.1 Import Commands
- [ ] `commands/import_cmd.py`:

  **Команды:**
  ```python
  @app.command()
  def okdesk(file: Path, wait: bool = False):
      """Импорт данных из OKDesk JSONL файла"""
      # 1. Валидация файла
      # 2. Вызов API
      # 3. Вывод статуса импорта
      # 4. Если wait=True, показать прогресс-бар и ждать завершения

  @app.command()
  def telegram(file: Path, wait: bool = False):
      """Импорт данных из Telegram JSON файла"""

  @app.command()
  def status(import_id: str):
      """Показать статус импорта по ID"""
      # Вывод таблицы со статусом, статистикой

  @app.command()
  def history(limit: int = 20):
      """История импортов"""
      # Таблица с последними импортами
  ```

- [ ] Интеграция с Rich:
  - Progress bar для wait режима
  - Таблица для истории импортов
  - Цветовые индикаторы статуса

### Этап 4: Команды pipeline обработки

#### 4.1 Pipeline Commands
- [ ] `commands/pipeline_cmd.py`:

  **Команды:**
  ```python
  @app.command()
  def process(
      batch_size: int = 100,
      device: str = "cpu",
      watch: bool = False
  ):
      """Запустить pipeline обработку необработанных issues"""
      # 1. Запуск обработки
      # 2. Если watch=True, показать прогресс
      # 3. Вывод финальной статистики

  @app.command()
  def status():
      """Показать статус pipeline"""
      # Таблица с количеством обработанных/необработанных
  ```

- [ ] Прогресс-бар с этапами:
  - Stage 1: Preprocessing
  - Stage 2: Embeddings
  - Stage 3: Vector DB Storage
  - Stage 4: Completion

### Этап 5: Команды кластеризации

#### 5.1 Clustering Commands
- [ ] `commands/clustering_cmd.py`:

  **Команды:**
  ```python
  @app.command()
  def run(
      method: str = "hdbscan",
      min_cluster_size: int = 5,
      use_llm: bool = False,
      wait: bool = True
  ):
      """Запустить кластеризацию"""
      # 1. Валидация method (hdbscan|kmeans)
      # 2. Запуск кластеризации
      # 3. Прогресс-бар если wait=True
      # 4. Вывод результатов

  @app.command()
  def list():
      """Список кластеров"""
      # Таблица: ID, название, размер, центроид

  @app.command()
  def show(cluster_id: str, limit: int = 10):
      """Показать issues в кластере"""
      # Таблица топ issues с расстояниями до центроида
  ```

### Этап 6: Команды поиска и просмотра

#### 6.1 Search Commands
- [ ] `commands/search_cmd.py`:

  **Команды:**
  ```python
  @app.command()
  def similar(
      query: str,
      top_k: int = 10,
      min_similarity: float = 0.7
  ):
      """Семантический поиск похожих issues"""
      # Таблица с issue_id, similarity score, snippet

  @app.command()
  def fulltext(
      query: str,
      limit: int = 50,
      offset: int = 0
  ):
      """Полнотекстовый поиск"""
      # Таблица результатов

  @app.command()
  def list_issues(
      status: str = None,
      source_id: str = None,
      limit: int = 50,
      offset: int = 0
  ):
      """Список issues с фильтрами"""
      # Таблица: ID, title, status, priority, created_at

  @app.command()
  def show_issue(issue_id: str):
      """Детали issue с messages"""
      # Форматированный вывод:
      # - Issue details (title, description, status)
      # - Messages timeline
      # - Cluster assignment (если есть)
  ```

### Этап 7: Команды статистики

#### 7.1 Stats Commands
- [ ] `commands/stats_cmd.py`:

  **Команды:**
  ```python
  @app.command()
  def processing():
      """Статистика обработки"""
      # Таблица:
      # - Total issues
      # - Processed
      # - Pending
      # - Success rate
      # - Average processing time

  @app.command()
  def clusters():
      """Статистика по кластерам"""
      # Таблица топ кластеров по размеру
      # График распределения (Rich bar chart)

  @app.command()
  def sources():
      """Статистика по источникам"""
      # Таблица: source name, total issues, processed

  @app.command()
  def timeline(
      from_date: str = None,
      to_date: str = None,
      granularity: str = "day"
  ):
      """Временная динамика"""
      # График количества issues по времени
      # ASCII график через Rich
  ```

### Этап 8: Команды экспорта

#### 8.1 Export Commands
- [ ] `commands/export_cmd.py`:

  **Команды:**
  ```python
  @app.command()
  def data(
      format: str = "csv",
      output: Path = None,
      status: str = None,
      cluster_id: str = None,
      from_date: str = None,
      to_date: str = None
  ):
      """Экспорт данных с фильтрами"""
      # 1. Валидация format (csv|json)
      # 2. Формирование фильтров
      # 3. Вызов API
      # 4. Сохранение файла
      # 5. Прогресс-бар для больших экспортов
  ```

### Этап 9: Форматирование вывода

#### 9.1 Output Infrastructure
- [ ] `output/styles.py`:
  - Цветовая схема (success=green, error=red, warning=yellow, info=blue)
  - Стили для разных типов данных
  - Theme customization

- [ ] `output/tables.py`:
  - `create_table(data, columns, title)` - Rich таблица
  - `create_issues_table(issues)` - специализированная таблица для issues
  - `create_clusters_table(clusters)` - таблица кластеров
  - `create_stats_table(stats)` - таблица статистики

- [ ] `output/progress.py`:
  - `create_progress_bar(description, total)` - Progress bar
  - `create_spinner(description)` - Spinner для ожидания
  - `update_progress(progress, current, description)` - обновление прогресса

- [ ] `output/formatters.py`:
  - `format_datetime(dt)` - форматирование дат
  - `format_status(status)` - цветной статус
  - `format_json_pretty(data)` - красивый JSON
  - `format_file_size(bytes)` - человекочитаемый размер

### Этап 10: Main Entry Point

#### 10.1 Main Application
- [ ] `main.py`:
  - Создание Typer app
  - Регистрация всех command groups:
    ```python
    app = typer.Typer(
        name="support-cli",
        help="CLI для системы анализа обращений",
        add_completion=True
    )

    app.add_typer(import_app, name="import")
    app.add_typer(pipeline_app, name="pipeline")
    app.add_typer(clustering_app, name="cluster")
    app.add_typer(search_app, name="search")
    app.add_typer(stats_app, name="stats")
    app.add_typer(export_app, name="export")
    ```

  - Global options:
    ```python
    @app.callback()
    def main(
        ctx: typer.Context,
        api_url: str = typer.Option(None, envvar="API_GATEWAY_URL"),
        verbose: bool = typer.Option(False, "--verbose", "-v"),
        quiet: bool = typer.Option(False, "--quiet", "-q"),
    ):
        """Global options"""
        # Инициализация APIClient
        # Настройка логирования
        # Сохранение в context
    ```

  - Error handling:
    - Try-catch вокруг команд
    - Понятные сообщения об ошибках
    - Exit codes (0=success, 1=error)

### Этап 11: Тестирование

#### 11.1 Unit Tests
- [ ] `tests/test_api_client.py`:
  - Моки httpx responses
  - Тесты всех методов APIClient
  - Тесты обработки ошибок

- [ ] `tests/test_commands/`:
  - Тесты каждой команды с мокированным APIClient
  - Тесты валидации входных данных
  - Тесты форматирования вывода

- [ ] `tests/test_formatters.py`:
  - Тесты форматирования дат, статусов, JSON
  - Тесты создания таблиц

#### 11.2 Integration Tests
- [ ] Тесты с реальным API (требуют запущенных сервисов):
  - Импорт тестового файла
  - Запуск pipeline
  - Поиск и просмотр
  - Статистика

### Этап 12: Docker и деплой

#### 12.1 Dockerfile
- [ ] Multi-stage build:
  ```dockerfile
  FROM python:3.12-slim AS builder
  # Установка uv
  # Копирование pyproject.toml
  # uv sync

  FROM python:3.12-slim AS runtime
  # Копирование виртуального окружения
  # Entrypoint: support-cli
  ```

#### 12.2 Docker Compose интеграция
- [ ] Добавить сервис в `docker-compose.yml`:
  ```yaml
  cli:
    build: ./services/cli
    container_name: support-cli
    depends_on:
      - api-gateway
    environment:
      API_GATEWAY_URL: http://api-gateway:8000
    volumes:
      - ./data:/app/data:ro
    networks:
      - support-network
    # Запуск как daemon или one-off команда
  ```

### Этап 13: Документация

#### 13.1 README.md
- [ ] Создать `services/cli/README.md`:
  - Описание CLI
  - Установка через uv: `uv pip install -e .`
  - Установка через Docker
  - Конфигурация (.env)
  - Примеры использования всех команд
  - Troubleshooting

#### 13.2 Help документация
- [ ] Добавить docstrings ко всем командам (Typer использует их для help)
- [ ] Примеры использования в help:
  ```python
  @app.command(
      help="Импорт данных из OKDesk",
      epilog="Пример: support-cli import okdesk /data/okdesk/out.jsonl --wait"
  )
  ```

---

## 4. Примеры использования

### 4.1 Импорт данных
```bash
# Импорт OKDesk
support-cli import okdesk /data/okdesk/out.jsonl --wait

# Импорт Telegram
support-cli import telegram /data/telegram/result.json

# Статус импорта
support-cli import status abc-123-def

# История импортов
support-cli import history --limit 10
```

### 4.2 Pipeline обработка
```bash
# Запуск обработки
support-cli pipeline process --batch-size 100 --watch

# Статус pipeline
support-cli pipeline status
```

### 4.3 Кластеризация
```bash
# Запуск с HDBSCAN
support-cli cluster run --method hdbscan --min-cluster-size 5

# С генерацией названий через LLM
support-cli cluster run --use-llm

# Список кластеров
support-cli cluster list

# Просмотр кластера
support-cli cluster show cluster-uuid --limit 20
```

### 4.4 Поиск
```bash
# Семантический поиск
support-cli search similar "проблема с оплатой" --top-k 10

# Полнотекстовый поиск
support-cli search fulltext "платеж не прошел"

# Список issues
support-cli search list-issues --status opened --limit 50

# Детали issue
support-cli search show-issue issue-uuid
```

### 4.5 Статистика
```bash
# Статистика обработки
support-cli stats processing

# Статистика кластеров
support-cli stats clusters

# По источникам
support-cli stats sources

# Временная динамика
support-cli stats timeline --from-date 2025-01-01 --to-date 2025-11-25 --granularity week
```

### 4.6 Экспорт
```bash
# Экспорт в CSV
support-cli export data --format csv --output /tmp/export.csv --status opened

# Экспорт кластера в JSON
support-cli export data --format json --cluster-id cluster-uuid --output /tmp/cluster.json
```

---

## 5. Критерии готовности (Definition of Done)

### Функциональность
- [ ] Все команды из спецификации реализованы
- [ ] API Client поддерживает все endpoints
- [ ] Обработка всех типов ошибок
- [ ] Прогресс-бары для длительных операций
- [ ] Валидация входных данных

### Качество кода
- [ ] Type hints везде
- [ ] Docstrings для всех публичных функций
- [ ] Unit тесты (coverage > 80%)
- [ ] Integration тесты для основных сценариев
- [ ] Линтеры проходят (ruff, mypy)

### UX
- [ ] Красивый вывод через Rich
- [ ] Цветовые индикаторы статусов
- [ ] Таблицы для структурированных данных
- [ ] Понятные сообщения об ошибках
- [ ] Help документация актуальна

### Деплой
- [ ] Dockerfile работает
- [ ] Docker Compose интеграция
- [ ] Можно установить через uv
- [ ] .env.example актуален
- [ ] README.md полный

### Документация
- [ ] README с примерами
- [ ] Help для всех команд
- [ ] Описание конфигурации
- [ ] Troubleshooting секция

---

## 6. Зависимости

**Требует:**
- ✅ Database Service (работает)
- ✅ Parser Service (работает)
- ✅ Analyzer Service (базовый функционал работает)
- 🔄 API Gateway (в разработке) - можно начинать, работая напрямую с сервисами

**Блокирует:**
- GUI Service (может использовать те же API endpoints)

---

## 7. Риски и митигация

### Риск 1: API Gateway не готов
**Митигация:** Реализовать поддержку как прямого подключения к сервисам, так и через Gateway:
```python
# config.py
class Settings:
    use_gateway: bool = True
    api_gateway_url: str = "http://localhost:8000"
    parser_url: str = "http://localhost:8001"
    analyzer_url: str = "http://localhost:8002"
```

### Риск 2: Новые endpoints Analyzer еще не реализованы
**Митигация:** Разработка CLI параллельно с Analyzer Extensions, тесная координация

### Риск 3: Сложность тестирования асинхронных операций
**Митигация:** Использовать pytest-asyncio, моки для httpx, time.sleep для эмуляции задержек

---

## 8. Временная оценка

- **Этап 1-2** (Инфраструктура + API Client): 1 день
- **Этап 3-8** (Все команды): 2 дня
- **Этап 9-10** (Форматирование + Main): 1 день
- **Этап 11** (Тестирование): 1 день
- **Этап 12-13** (Docker + Документация): 0.5 дня

**Итого: 5-6 дней**

---

## 9. Следующие шаги

1. ✅ Создать детальный план (этот документ)
2. ⏭️ Получить approval от пользователя
3. ⏭️ Начать реализацию поэтапно с коммитами `[06-cli-service] description`
4. ⏭️ После завершения создать changelog в `changelog.d/06-cli-service.md`
