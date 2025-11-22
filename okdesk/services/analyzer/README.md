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

## Конфигурация

Переменные окружения (`.env`):

```env
# Database
DATABASE_URL=postgresql+asyncpg://support:support@localhost:15432/support

# ChromaDB
CHROMADB_HOST=localhost
CHROMADB_PORT=8100
CHROMADB_COLLECTION=support_issues

# ML Models
EMBEDDING_MODEL=intfloat/multilingual-e5-large
EMBEDDING_DIM=1024
MAX_SEQ_LENGTH=512
DEVICE=cpu

# Pipeline
BATCH_SIZE=100
EMBEDDING_BATCH_SIZE=32
MAX_RETRIES=3
RETRY_DELAY=5

# Clustering
CLUSTERING_METHOD=hdbscan
HDBSCAN_MIN_CLUSTER_SIZE=5
HDBSCAN_MIN_SAMPLES=3
```

## Запуск

### Локально

```bash
# Запустить сервер
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8002

# API документация
open http://localhost:8002/docs
```

### Docker

```bash
# Из корня проекта
docker compose up analyzer
```

## API Endpoints

### Pipeline Management

- `POST /api/v1/analyzer/process` - Обработка батча issues
- `POST /api/v1/analyzer/resume` - Возобновление после сбоя
- `POST /api/v1/analyzer/reprocess/{issue_id}` - Переобработка issue

### Clustering

- `POST /api/v1/analyzer/cluster` - Запуск кластеризации
- `GET /api/v1/analyzer/clusters` - Список кластеров
- `GET /api/v1/analyzer/clusters/{id}` - Детали кластера

### Search

- `POST /api/v1/analyzer/search/similar` - Семантический поиск

### Statistics

- `GET /api/v1/analyzer/stats` - Статистика обработки

## Примеры использования

### Обработка issues

```bash
curl -X POST http://localhost:8002/api/v1/analyzer/process \
  -H "Content-Type: application/json" \
  -d '{"batch_size": 100}'
```

### Кластеризация

```bash
curl -X POST http://localhost:8002/api/v1/analyzer/cluster \
  -H "Content-Type: application/json" \
  -d '{"method": "hdbscan", "use_llm": false}'
```

### Поиск похожих issues

```bash
curl -X POST http://localhost:8002/api/v1/analyzer/search/similar \
  -H "Content-Type: application/json" \
  -d '{"issue_id": "uuid-here", "top_k": 10}'
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

Примерные метрики (CPU):

- **Preprocessing:** ~100-200 issues/sec
- **Embeddings:** ~50-100 issues/sec (bottleneck)
- **Vector DB:** ~500-1000 issues/sec

**Итого:** 10000 issues обрабатываются за ~3-6 минут

Для ускорения:
- Использовать GPU (`DEVICE=cuda`)
- Увеличить `EMBEDDING_BATCH_SIZE`
- Использовать более легкую модель

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

### Медленная обработка

```bash
# Уменьшить batch size
export BATCH_SIZE=50

# Использовать более легкую модель
export EMBEDDING_MODEL=cointegrated/rubert-tiny2
```

## License

Internal project
