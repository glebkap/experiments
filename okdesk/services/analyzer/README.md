# Analyzer Service

Pipeline-обработка issues с генерацией embeddings и кластеризацией.

## Описание

Analyzer Service выполняет многоэтапную обработку issues из системы поддержки:

1. **Stage 0: Fetch** - Получение необработанных issues из PostgreSQL
2. **Stage 1: Preprocessing** - Очистка HTML, лемматизация, нормализация текста
3. **Stage 2: Embeddings** - Генерация векторных представлений через SentenceTransformers
4. **Stage 3: Vector DB** - Сохранение embeddings в ChromaDB
5. **Stage 4: Complete** - Финализация обработки

Дополнительно предоставляет:
- **Кластеризацию** - группировка похожих issues (HDBSCAN/K-means)
- **Семантический поиск** - поиск похожих обращений

## Архитектура

Сервис построен по принципам **Domain-Driven Design (DDD)**:

```
src/
├── domain/           # Бизнес-логика
│   ├── models/       # Entities, Value Objects
│   ├── repositories/ # Repository interfaces
│   └── services/     # Domain services
├── application/      # Use cases
│   ├── pipeline/     # Pipeline stages
│   ├── use_cases/    # Business use cases
│   └── dto/          # Data Transfer Objects
├── infrastructure/   # Внешние зависимости
│   ├── persistence/  # PostgreSQL repositories
│   ├── ml/           # ML models (SentenceTransformers)
│   └── vectordb/     # ChromaDB client
└── interfaces/       # API endpoints
    └── api/          # FastAPI routes
```

## Технологии

- **Python 3.12** + uv
- **FastAPI** - REST API
- **SQLAlchemy 2.0** - ORM для PostgreSQL
- **SentenceTransformers** - генерация embeddings
  - Модель: `intfloat/multilingual-e5-large` (1024d)
- **ChromaDB** - векторное хранилище
- **HDBSCAN** / **K-means** - кластеризация
- **BeautifulSoup4** - очистка HTML
- **pymorphy2** - лемматизация русского языка

## Установка

```bash
# Создать виртуальное окружение и установить зависимости
uv sync

# Активировать окружение
source .venv/bin/activate
```

## Требования

Перед запуском необходимо:

1. **PostgreSQL** с данными issues (порт 5432)
2. **ChromaDB** для хранения embeddings (порт 8100)

```bash
# Запуск PostgreSQL (из корня проекта)
cd db
make run

# Запуск ChromaDB (из корня проекта)
cd db_vector
docker compose up -d
```

## Конфигурация

Переменные окружения (`.env`):

```env
# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/postgres

# ChromaDB
CHROMADB_HOST=localhost
CHROMADB_PORT=8100
CHROMADB_COLLECTION=support_issues

# ML Models
EMBEDDING_MODEL=intfloat/multilingual-e5-large  # Модель для embeddings (размерность определяется автоматически)
MAX_SEQ_LENGTH=512                               # Макс. длина последовательности
DEVICE=auto  # auto (авто-определение), cpu, cuda (NVIDIA GPU), mps (Apple Silicon)

# Processing Manager
BATCH_SIZE=100                     # Количество issues в одном батче
EMBEDDING_BATCH_SIZE=32            # Batch size для генерации embeddings
POLL_INTERVAL_SECONDS=1.0          # Интервал опроса новых issues
AUTO_START_PROCESSING=false        # Авто-запуск обработки при старте

# Pipeline
MAX_RETRIES=3
RETRY_DELAY=5

# Clustering
CLUSTERING_METHOD=hdbscan
HDBSCAN_MIN_CLUSTER_SIZE=5
HDBSCAN_MIN_SAMPLES=3
```

## Запуск

### 1. Установка зависимостей

```bash
# Установить uv (если еще не установлен)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Синхронизировать зависимости
uv sync
```

### 2. Запуск сервиса

```bash
# Вариант 1: Через uv run (рекомендуется для разработки)
uv run python -m src.main

# Вариант 2: Через uvicorn с auto-reload
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8002
```

Сервис будет доступен на `http://localhost:8002`

### 3. Проверка работы

```bash
# Health check
curl http://localhost:8002/health

# Статус сервиса
curl http://localhost:8002/api/v1/analyzer/status | python -m json.tool

# Статус обработки
curl http://localhost:8002/api/v1/analyzer/processing/status | python -m json.tool
```

### 4. API документация

Swagger UI доступен по адресу: `http://localhost:8002/docs`

## API Endpoints

### Processing Manager (Управление фоновой обработкой)

- `POST /api/v1/analyzer/processing/start` - Запустить фоновую обработку
- `POST /api/v1/analyzer/processing/pause` - Приостановить обработку
- `POST /api/v1/analyzer/processing/resume` - Возобновить обработку
- `POST /api/v1/analyzer/processing/stop` - Остановить обработку
- `GET /api/v1/analyzer/processing/status` - Текущий статус обработки
- `POST /api/v1/analyzer/processing/process-batch` - Обработать один батч вручную

### Issue Management

- `POST /api/v1/analyzer/issues/{issue_id}/reprocess` - Переобработать конкретный issue

### Clustering

- `POST /api/v1/analyzer/clustering/run` - Запуск кластеризации
- `GET /api/v1/analyzer/clustering/info` - Информация о кластерах

### Search

- `POST /api/v1/analyzer/search/similar` - Семантический поиск похожих issues

### Service Status

- `GET /api/v1/analyzer/status` - Общая статистика сервиса
- `GET /health` - Health check endpoint

## Примеры использования

### Запуск фоновой обработки

```bash
# Запустить обработку
curl -X POST http://localhost:8002/api/v1/analyzer/processing/start

# Проверить статус
curl http://localhost:8002/api/v1/analyzer/processing/status | python -m json.tool

# Остановить обработку
curl -X POST http://localhost:8002/api/v1/analyzer/processing/stop
```

**Пример вывода статуса:**
```json
{
  "state": "running",
  "batch_size": 100,
  "poll_interval_seconds": 1.0,
  "total_processed": 600,
  "total_batches": 6,
  "started_at": "2025-11-22T21:32:37.736035",
  "last_batch_at": "2025-11-22T21:43:02.894393",
  "uptime_seconds": 643.12
}
```

### Обработка одного батча вручную

```bash
curl -X POST http://localhost:8002/api/v1/analyzer/processing/process-batch
```

### Переобработка конкретного issue

```bash
curl -X POST http://localhost:8002/api/v1/analyzer/issues/{issue-uuid}/reprocess
```

### Кластеризация

```bash
curl -X POST http://localhost:8002/api/v1/analyzer/clustering/run \
  -H "Content-Type: application/json" \
  -d '{
    "method": "hdbscan",
    "min_cluster_size": 5,
    "min_samples": 3
  }'
```

### Поиск похожих issues

```bash
# По ID существующего issue
curl -X POST http://localhost:8002/api/v1/analyzer/search/similar \
  -H "Content-Type: application/json" \
  -d '{
    "issue_id": "uuid-here",
    "top_k": 10,
    "min_similarity": 0.7
  }'

# По текстовому запросу
curl -X POST http://localhost:8002/api/v1/analyzer/search/similar \
  -H "Content-Type: application/json" \
  -d '{
    "query": "проблема с доступом к сайту",
    "top_k": 10,
    "min_similarity": 0.7
  }'
```

### Статистика сервиса

```bash
curl http://localhost:8002/api/v1/analyzer/status | python -m json.tool
```

**Пример вывода:**
```json
{
  "total_issues": 27402,
  "unprocessed_issues": 26802,
  "processed_issues": 600,
  "total_embeddings": 600,
  "total_clusters": 0
}
```

## Разработка

### Тестирование

```bash
# Unit тесты
uv run pytest tests/unit -v

# Integration тесты
uv run pytest tests/integration -v

# С покрытием
uv run pytest --cov=src --cov-report=html
```

### Структура кода

- `domain/` - бизнес-логика, не зависит от внешних библиотек
- `application/` - use cases, pipeline orchestration
- `infrastructure/` - адаптеры для внешних систем
- `interfaces/` - HTTP API endpoints

## Производительность

### Метрики по устройствам

**CPU (интегрированный Intel/AMD):**
- Preprocessing: ~100-200 issues/sec
- Embeddings: ~50-100 issues/sec (bottleneck)
- Vector DB: ~500-1000 issues/sec
- **Итого:** 10000 issues за ~3-6 минут

**CUDA (NVIDIA GPU, например RTX 3060):**
- Preprocessing: ~100-200 issues/sec
- Embeddings: ~300-500 issues/sec (ускорение в 5-10x)
- Vector DB: ~500-1000 issues/sec
- **Итого:** 10000 issues за ~1-2 минуты

**MPS (Apple Silicon M1/M2):**
- Preprocessing: ~100-200 issues/sec
- Embeddings: ~200-400 issues/sec (ускорение в 3-5x)
- Vector DB: ~500-1000 issues/sec
- **Итого:** 10000 issues за ~1.5-2.5 минуты

### Оптимизация производительности

1. **Использовать GPU (рекомендуется):**
   ```bash
   # Авто-определение лучшего device
   export DEVICE=auto

   # Или явно указать
   export DEVICE=cuda     # для NVIDIA GPU
   export DEVICE=mps      # для Apple Silicon
   ```

2. **Настроить batch size:**
   ```bash
   # Увеличить для GPU (больше VRAM)
   export EMBEDDING_BATCH_SIZE=64   # или 128 для мощных GPU

   # Уменьшить для CPU или слабых GPU
   export EMBEDDING_BATCH_SIZE=16
   ```

3. **Выбрать модель под задачу:**
   ```bash
   # Максимальное качество (1024d, ~2GB VRAM)
   export EMBEDDING_MODEL=intfloat/multilingual-e5-large

   # Баланс качество/скорость (768d, ~1GB VRAM)
   export EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2

   # Максимальная скорость (312d, ~200MB VRAM)
   export EMBEDDING_MODEL=cointegrated/rubert-tiny2
   ```

### Требования к GPU

| Модель | VRAM (batch=32) | VRAM (batch=64) | Рекомендуемый GPU |
|--------|-----------------|-----------------|-------------------|
| multilingual-e5-large (1024d) | ~3-4 GB | ~6-8 GB | RTX 3060 6GB+ |
| mpnet-base-v2 (768d) | ~2-3 GB | ~4-5 GB | GTX 1660 6GB+ |
| rubert-tiny2 (312d) | ~1-2 GB | ~2-3 GB | GTX 1050 Ti 4GB+ |

**Примечание:** Для Apple Silicon рекомендуется использовать unified memory (от 8GB RAM)

## Отладка и логирование

### Уровни логирования

Сервис поддерживает различные уровни логирования для отладки:

```bash
# INFO (по умолчанию) - основные события
export LOG_LEVEL=INFO

# DEBUG - детальная информация о каждом issue
export LOG_LEVEL=DEBUG

# WARNING - только предупреждения и ошибки
export LOG_LEVEL=WARNING
```

### Настройки debug-логирования

Дополнительные настройки для управления объемом debug-логов:

```bash
# Логировать полный текст issues (default: true)
export DEBUG_LOG_ISSUE_CONTENT=true

# Максимальная длина текста для логирования (0 = без ограничений)
export DEBUG_LOG_MAX_CONTENT_LENGTH=500

# Количество issues для детального логирования (default: 3)
export DEBUG_LOG_SAMPLE_SIZE=3
```

**Рекомендации:**
- Для production: `DEBUG_LOG_ISSUE_CONTENT=false` (логировать только превью)
- Для отладки конкретных issues: `DEBUG_LOG_SAMPLE_SIZE=1`, `DEBUG_LOG_MAX_CONTENT_LENGTH=0`
- Для анализа больших батчей: `DEBUG_LOG_SAMPLE_SIZE=10`, `DEBUG_LOG_MAX_CONTENT_LENGTH=200`

### Отслеживание обработки issues

При уровне **INFO** вы увидите:
- Количество загруженных issues
- Статус каждого stage pipeline
- Метрики производительности (время, скорость)
- Финальную статистику

При уровне **DEBUG** добавляется:
- **Issue IDs** первых 3-5 issues
- Превью контента (title/description)
- Длины текстов до и после preprocessing
- Детали embeddings (norms, shape, dtype)
- Информация о сохранении в ChromaDB

**Пример лога при DEBUG (с DEBUG_LOG_ISSUE_CONTENT=true):**

```
======================================================================
[ProcessIssuesBatch] Starting batch processing
[ProcessIssuesBatch] Batch size: 100, Device: cuda
======================================================================
[ProcessIssuesBatch] Pipeline has 5 stages
[Stage 0] Fetching up to 100 unprocessed issues...
[Stage 0] Fetched 100 unprocessed issues
[Stage 0] Issue IDs (100 total):
    abc-123-def, abc-456-ghi, abc-789-jkl, ...
[Stage 0] Detailed info for first 3 issues:

================================================================================
Issue abc-123-def (1/3)
================================================================================
Status: opened
--------------------------------------------------------------------------------
Title:
    Не работает оплата картой на сайте
--------------------------------------------------------------------------------
Description (length=234):
    Добрый день! При попытке оплатить заказ картой Visa на сайте
    shop.example.com выдает ошибку "Платеж отклонен". Пробовал разные
    карты - результат тот же. Через мобильное приложение оплата проходит
    нормально. Браузер Chrome последней версии.
================================================================================

[Stage 1] Preprocessing 100 issues...

================================================================================
Preprocessed Issue abc-123-def (1/3)
================================================================================
Original lengths: title=39, description=234
Preprocessed length: 198
Compression: 72.5%
--------------------------------------------------------------------------------
Preprocessed content:
    работать оплата карта сайт день попытка оплатить заказ карта visa
    сайт выдавать ошибка платеж отклонить пробовать разный карта
    результат мобильный приложение оплата проходить нормально браузер
    chrome последний версия
================================================================================

[Stage 1] Saving 100 preprocessed issues to database...
[Stage 1] Preprocessed 100 issues (skipped 0 with empty content)

[Stage 2] Generating embeddings for 100 texts (batch_size=32)...
[Stage 2] Model: intfloat/multilingual-e5-large, Dimension: 1024
[Stage 2] Generated 100 embeddings (dim=1024, time=12.34s, rate=8.1 issues/sec)
[Stage 2] Embedding norms: min=0.998, max=1.002, mean=1.000

[Stage 3] Storing 100 embeddings in ChromaDB...
[Stage 3] First 3 issue IDs: ['abc123', 'def456', 'ghi789']
[Stage 3] Embedding shape: (100, 1024), dtype: float32
[Stage 3] Stored 100 embeddings successfully (time=0.45s, rate=222.2 issues/sec)
[Stage 3] Total embeddings in ChromaDB: 600

[Stage 4] ========== Pipeline Execution Completed ==========
[Stage 4] Final Statistics:
  - Issues fetched: 100
  - Issues preprocessed: 100
  - Issues skipped (empty): 0
  - Embeddings generated: 100
  - Embeddings stored: 100
  - Embedding dimension: 1024
  - Embedding time: 12.34s
  - VectorDB time: 0.45s
  - Overall rate: 7.8 issues/sec
[Stage 4] ================================================

======================================================================
[ProcessIssuesBatch] Pipeline finished
[ProcessIssuesBatch] Success: True
[ProcessIssuesBatch] Processed: 100 issues
[ProcessIssuesBatch] Duration: 13.12s
======================================================================
```

### Мониторинг в реальном времени

```bash
# Запустить сервис с DEBUG логированием
export LOG_LEVEL=DEBUG
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8002

# В другом терминале - следить за логами
tail -f logs/analyzer.log | grep -E "\[Stage|ProcessIssuesBatch\]"

# Или использовать jq для форматирования (если логи в JSON)
tail -f logs/analyzer.log | jq -r '.message'
```

### Трейсинг конкретного issue

Чтобы отследить обработку конкретного issue, найдите его ID в логах:

```bash
# Найти все упоминания конкретного issue
grep "abc-123-def-456" logs/analyzer.log

# Пример вывода:
# [Stage 0] First 5 issue IDs: abc-123-def-456, ...
# [Stage 1] Processing issue 1/100: ID=abc-123-def-456, title_len=45
# [Stage 1] Issue abc-123-def-456 preprocessed: content_len=198
# [Stage 3] First 3 issue IDs: ['abc-123-def-456', ...]
```

### Debugging через API

```bash
# Получить статус обработки
curl http://localhost:8002/api/v1/analyzer/status | python -m json.tool

# Обработать 1 issue для отладки
curl -X POST http://localhost:8002/api/v1/analyzer/processing/process-batch \
  -H "Content-Type: application/json" \
  -d '{"batch_size": 1}'

# Переобработать конкретный issue
curl -X POST http://localhost:8002/api/v1/analyzer/issues/{issue-id}/reprocess
```

## Troubleshooting

### Модель embeddings не загружается

```bash
# Скачать модель вручную
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-large')"
```

### ChromaDB недоступен

```bash
# Проверить статус
cd ../../db_vector
make health
```

### CUDA не доступен (ошибка "CUDA is not available")

**Проблема:** Сервис не может использовать GPU, хотя он установлен.

**Решения:**

1. **Проверить установку CUDA:**
   ```bash
   # Проверить наличие NVIDIA GPU
   nvidia-smi

   # Проверить CUDA в PyTorch
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

2. **Переустановить PyTorch с CUDA:**
   ```bash
   # Для CUDA 11.8
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

   # Для CUDA 12.1
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Использовать CPU как fallback:**
   ```bash
   export DEVICE=cpu
   ```

### MPS не доступен (Apple Silicon)

**Проблема:** MPS недоступен на macOS.

**Требования:**
- macOS 12.3 или новее
- Apple Silicon (M1/M2/M3)

**Решение:**
```bash
# Обновить macOS или использовать CPU
export DEVICE=cpu
```

### Out of Memory (OOM) на GPU

**Проблема:** GPU не хватает памяти при обработке.

**Решения:**

1. **Уменьшить batch size:**
   ```bash
   export EMBEDDING_BATCH_SIZE=16  # или даже 8
   ```

2. **Использовать более легкую модель:**
   ```bash
   export EMBEDDING_MODEL=cointegrated/rubert-tiny2
   ```

3. **Очистить GPU память:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Медленная обработка

```bash
# Использовать авто-определение лучшего device
export DEVICE=auto

# Увеличить batch size (если есть память)
export EMBEDDING_BATCH_SIZE=64

# Уменьшить общий batch size (если упираетесь в RAM)
export BATCH_SIZE=50

# Использовать более легкую модель
export EMBEDDING_MODEL=cointegrated/rubert-tiny2
```

## License

Internal project
