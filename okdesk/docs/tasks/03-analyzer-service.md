# Analyzer Service - Декомпозиция задач

**Сервис:** support-analyzer
**Приоритет:** Критический (основной функционал)
**Технологии:** Python 3.12, FastAPI, SentenceTransformers, ChromaDB, HDBSCAN, scikit-learn, uv
**Архитектура:** DDD (Domain-Driven Design) + Pipeline Processing

---

## Обзор

Analyzer Service выполняет **пакетную обработку issues** через многоэтапный pipeline. Каждое issue проходит последовательную обработку: предобработка текста → генерация эмбеддингов → сохранение в векторную БД. Промежуточные результаты сохраняются в PostgreSQL, что позволяет возобновить обработку после сбоев.

Кластеризация выполняется **отдельно** после того, как все issues обработаны, через специальную команду.

### Ключевые принципы

- **Issue-centric processing** - обрабатываем целые issues (title + description), а не отдельные messages
- **Pipeline architecture** - 4 этапа обработки + отдельная кластеризация
- **Масштабируемость** - способность обработать 10000+ issues
- **Fault tolerance** - возможность возобновления после сбоев
- **Batch processing** - обработка N issues за раз (по умолчанию 100)
- **Deferred clustering** - кластеризация выполняется отдельной командой

---

## Архитектура Pipeline

### Общая схема обработки

```
┌──────────────────────────────────────────────────────────────┐
│                    Analyzer Pipeline                          │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Stage 0: Data Fetching                                       │
│  ├─ Получить N необработанных issues                         │
│  └─ Загрузить связанные messages (для будущего расширения)  │
│                                                               │
│  Stage 1: Preprocessing                                       │
│  ├─ Извлечь issue.title и issue.description                  │
│  ├─ Очистка HTML/XML тегов                                   │
│  ├─ Нормализация текста (lower, spaces, punctuation)        │
│  ├─ Удаление стоп-слов и шаблонных фраз                     │
│  ├─ Лемматизация (pymorphy2)                                │
│  ├─ Токенизация спец. паттернов [URL], [EMAIL], etc         │
│  └─ Сохранение в preprocessed_issues                         │
│                                                               │
│  Stage 2: Embedding Generation                                │
│  ├─ Загрузка SentenceTransformer модели                      │
│  ├─ Генерация эмбеддингов для preprocessed content          │
│  └─ L2 нормализация векторов                                 │
│                                                               │
│  Stage 3: Vector DB Storage                                   │
│  ├─ Сохранение embeddings в ChromaDB                         │
│  ├─ Связь только через issue.id                              │
│  └─ Индексирование для семантического поиска                 │
│                                                               │
│  Stage 4: Completion                                          │
│  └─ Логирование завершения обработки                         │
│                                                               │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│           Clustering (отдельная команда)                      │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Загрузка всех embeddings из ChromaDB                     │
│  2. Применение HDBSCAN/K-means                               │
│  3. Определение оптимального числа кластеров                 │
│  4. Присвоение cluster_id каждому issue                      │
│  5. Опционально: определение названий через LLM              │
│  6. Сохранение результатов в clusters и message_clusters     │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Схема БД для обработки

Таблица `preprocessed_issues`:

```sql
CREATE TABLE preprocessed_issues (
  id UUID PRIMARY KEY REFERENCES issues(id),
  content TEXT NOT NULL,  -- результат препроцессинга (title + description)
  processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_preprocessed_issues_processed ON preprocessed_issues(processed_at);
CREATE INDEX idx_preprocessed_issues_text ON preprocessed_issues
  USING gin(to_tsvector('russian', content));
```

**Логика определения обработанности:**
- Issue считается необработанным, если его нет в `preprocessed_issues`
- Запрос: `SELECT * FROM issues WHERE id NOT IN (SELECT id FROM preprocessed_issues)`

---

## DDD Структура

```
services/analyzer/
├── src/
│   ├── domain/                     # Domain layer
│   │   ├── models/
│   │   │   ├── issue.py            # Issue entity
│   │   │   ├── message.py          # Message entity
│   │   │   ├── preprocessed_issue.py  # PreprocessedIssue value object
│   │   │   ├── embedding.py        # Embedding value object
│   │   │   └── cluster.py          # Cluster entity
│   │   ├── repositories/
│   │   │   ├── issue_repository.py
│   │   │   ├── message_repository.py
│   │   │   ├── preprocessed_issue_repository.py
│   │   │   └── cluster_repository.py
│   │   └── services/
│   │       ├── text_preprocessor.py        # Stage 1 логика
│   │       ├── embedding_generator.py      # Stage 2 логика
│   │       ├── vector_db_service.py        # Stage 3 логика
│   │       └── clustering_service.py       # Кластеризация
│   │
│   ├── application/                # Application layer
│   │   ├── pipeline/
│   │   │   ├── pipeline_config.py          # Конфигурация pipeline
│   │   │   ├── pipeline_executor.py        # Главный orchestrator
│   │   │   ├── stages/
│   │   │   │   ├── base_stage.py           # Базовый класс Stage
│   │   │   │   ├── stage_0_fetch.py        # Data fetching
│   │   │   │   ├── stage_1_preprocess.py   # Preprocessing
│   │   │   │   ├── stage_2_embed.py        # Embedding generation
│   │   │   │   ├── stage_3_vectordb.py     # Vector DB storage
│   │   │   │   └── stage_4_complete.py     # Completion
│   │   │   └── pipeline_context.py         # Shared context
│   │   ├── use_cases/
│   │   │   ├── process_issues_batch.py     # Основной use case
│   │   │   ├── resume_processing.py        # Возобновление после сбоя
│   │   │   ├── reprocess_issue.py          # Переобработка issue
│   │   │   └── cluster_all_issues.py       # Кластеризация
│   │   └── dto/
│   │       ├── issue_batch_dto.py
│   │       ├── processing_result_dto.py
│   │       ├── pipeline_stats_dto.py
│   │       └── clustering_result_dto.py
│   │
│   ├── infrastructure/              # Infrastructure layer
│   │   ├── persistence/
│   │   │   ├── postgres/
│   │   │   │   ├── issue_repository_impl.py
│   │   │   │   ├── message_repository_impl.py
│   │   │   │   ├── preprocessed_issue_repository_impl.py
│   │   │   │   └── cluster_repository_impl.py
│   │   │   ├── models.py                   # SQLAlchemy models
│   │   │   └── database.py
│   │   ├── ml/
│   │   │   ├── sentence_transformer.py     # SentenceTransformers wrapper
│   │   │   ├── clustering_algorithms.py    # HDBSCAN, K-means
│   │   │   └── text_normalizer.py          # pymorphy2, regex patterns
│   │   ├── vectordb/
│   │   │   ├── chromadb_client.py          # ChromaDB integration
│   │   │   └── collection_manager.py       # Collection CRUD
│   │   └── llm/
│   │       └── cluster_labeling.py         # LLM для названий кластеров (опционально)
│   │
│   ├── interfaces/                  # Interface adapters
│   │   └── api/
│   │       ├── routes.py
│   │       └── schemas.py
│   │
│   ├── config.py
│   ├── dependencies.py
│   └── main.py
│
├── tests/
│   ├── unit/
│   │   ├── domain/
│   │   ├── application/
│   │   └── infrastructure/
│   ├── integration/
│   └── fixtures/
│
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Подробное описание Pipeline Stages

### Stage 0: Data Fetching

**Цель:** Получить N необработанных issues из БД

**Входные данные:**
- `batch_size` (default: 100)

**Процесс:**
1. Запросить issues, которых нет в `preprocessed_issues`
2. Опционально загрузить messages для каждого issue (для будущего расширения функционала)

**SQL запрос:**
```sql
SELECT * FROM issues
WHERE id NOT IN (SELECT id FROM preprocessed_issues)
LIMIT :batch_size;
```

**Выходные данные:**
- Список issues в `PipelineContext`
- Статистика: количество issues и messages

**Класс:** `Stage0FetchIssues`

---

### Stage 1: Preprocessing

**Цель:** Очистить и нормализовать текст issue (title + description)

**Входные данные:**
- Список issues из Stage 0

**Процесс предобработки:**

#### 1.1 Базовая очистка
- Удаление HTML/XML тегов (BeautifulSoup)
- Приведение к нижнему регистру
- Удаление лишних пробелов, табуляций, переносов
- Удаление спецсимволов (оставляем буквы, цифры, базовую пунктуацию)

#### 1.2 Нормализация паттернов
- URL → `[URL]`
- Email → `[EMAIL]`
- Телефон → `[PHONE]`
- Даты к единому формату
- Числительные → цифры

#### 1.3 Доменная специфика
- Нормализация версий ПО (v1.2.3 → версия_1.2.3)
- Нормализация кодов ошибок
- Удаление шаблонных фраз поддержки ("здравствуйте", "с уважением")

#### 1.4 Лингвистическая обработка
- Лемматизация через pymorphy2 (приведение к нормальной форме)
- Сохранение токенов типа [URL], [EMAIL] без лемматизации

#### 1.5 Объединение полей
- Извлечение `issue.title` и `issue.description`
- Предобработка каждого поля отдельно
- Объединение: `"{title_processed}\n\n{description_processed}"`
- Заголовок идет первым (более важный для семантики)

**Сохранение результатов:**
- Batch-сохранение в `preprocessed_issues` (id, content, processed_at)

**Выходные данные:**
- Записи в `preprocessed_issues`
- Список `PreprocessedIssue` в `PipelineContext`

**Класс:** `Stage1Preprocess`

---

### Stage 2: Embedding Generation

**Цель:** Генерация векторных представлений для preprocessed issues

**Входные данные:**
- Preprocessed issues из Stage 1

**Процесс:**

#### 2.1 Загрузка модели
- SentenceTransformer модель
- max_seq_length = 512
- Рекомендуемые модели:
  - `intfloat/multilingual-e5-large` - лучшее качество для русского (1024d)
  - `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` - баланс (768d)
  - `cointegrated/rubert-tiny2` - быстрая модель (312d)

#### 2.2 Генерация embeddings
- Batch обработка (по умолчанию 32 текста за раз)
- L2 нормализация векторов
- Прогресс-бар для длительных операций

**Выходные данные:**
- Embeddings НЕ сохраняются в PostgreSQL
- Embeddings передаются в Stage 3 через `PipelineContext`
- Массив numpy (n_issues, embedding_dim)

**Класс:** `Stage2GenerateEmbeddings`

---

### Stage 3: Vector DB Storage

**Цель:** Сохранить embeddings в ChromaDB для семантического поиска

**Входные данные:**
- Embeddings из Stage 2
- Issue IDs и preprocessed content

**Процесс:**

#### 3.1 ChromaDB Setup
- Создание/подключение к коллекции
- Метрика: cosine similarity
- Персистентное хранилище

#### 3.2 Сохранение
- Идентификатор: `issue.id` (UUID как строка)
- Embedding: вектор из Stage 2
- Document: preprocessed content (для отображения результатов)
- **Метаданные НЕ сохраняются** - связь только через issue.id

#### 3.3 Функциональность поиска
- Семантический поиск похожих issues
- Возврат списка issue.id по убыванию релевантности
- Top-K результатов

**Выходные данные:**
- Embeddings сохранены в ChromaDB
- Готовность к семантическому поиску

**Класс:** `Stage3VectorDBStorage`

---

### Stage 4: Completion

**Цель:** Завершить pipeline обработки

**Процесс:**
1. Логирование успешной обработки
2. Обновление статистики в `PipelineContext`

**Примечание:**
Факт обработки фиксируется через наличие записи в `preprocessed_issues` (Stage 1). Отдельная таблица для отслеживания статусов не требуется.

**Выходные данные:**
- Логи завершения
- Финальная статистика pipeline

**Класс:** `Stage4Complete`

---

## Кластеризация (отдельная команда)

**Цель:** Группировка схожих issues в кластеры

**Когда запускается:** После обработки всех или большинства issues через pipeline

### Процесс кластеризации

#### 1. Загрузка данных
- Получение всех embeddings из ChromaDB
- Формирование матрицы (n_issues, embedding_dim)

#### 2. Применение алгоритма
- **HDBSCAN** (по умолчанию):
  - Автоматическое определение числа кластеров
  - Обработка шума (outliers)
  - Параметры: min_cluster_size, min_samples
  - Метрика: cosine

- **K-means** (альтернатива):
  - Определение оптимального K через silhouette score
  - Фиксированное число кластеров
  - Быстрее HDBSCAN

#### 3. Создание кластеров
- Создание записей в таблице `clusters`
- Вычисление centroid (среднее embedding)
- Подсчет размера кластера

#### 4. Назначение issues
- Связывание issues с кластерами через `message_clusters`
- Вычисление расстояния до центроида (cosine distance)
- Сортировка по расстоянию (близость к центру)

#### 5. Опционально: LLM labeling
- Выбор репрезентативных issues из кластера
- Отправка в LLM для анализа
- Генерация названия и описания кластера
- Сохранение в `clusters.name` и `clusters.description`

### Схема БД для кластеров

```sql
CREATE TABLE clusters (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  cluster_label INT NOT NULL UNIQUE,
  name TEXT,
  description TEXT,
  centroid_embedding vector(1024),
  size INT DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE message_clusters (
  message_id UUID REFERENCES messages(id),
  cluster_id UUID REFERENCES clusters(id),
  distance_to_centroid FLOAT,
  assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (message_id, cluster_id)
);

CREATE INDEX idx_clusters_size ON clusters(size DESC);
CREATE INDEX idx_message_clusters_cluster_id ON message_clusters(cluster_id);
CREATE INDEX idx_message_clusters_distance ON message_clusters(distance_to_centroid);
```

**Use Case:** `ClusterAllIssuesUseCase`

---

## Pipeline Executor

**Главный orchestrator для выполнения pipeline**

### Конфигурация

**PipelineConfig:**
- `batch_size` - размер батча (default: 100)
- `embedding_model` - модель для embeddings
- `preprocessing_config` - настройки предобработки
- `max_retries` - количество повторов при ошибке
- `retry_delay_seconds` - задержка между повторами

**PipelineContext:**
- Shared state между всеми stages
- Содержит: issues, preprocessed_issues, embeddings, статистика
- Передается через все stages последовательно

### Выполнение

1. Инициализация context с конфигурацией
2. Последовательное выполнение всех stages
3. Передача context от stage к stage
4. Обработка ошибок с retry логикой
5. Возврат результата (`PipelineResult`)

### Обработка ошибок

- Логирование всех ошибок
- Retry механизм (настраиваемое количество попыток)
- Возможность продолжить с конкретного stage
- Транзакционность на уровне batch

**Класс:** `PipelineExecutor`

---

## Use Cases

### 1. ProcessIssuesBatch

**Назначение:** Обработка батча необработанных issues

**Параметры:**
- `batch_size` - количество issues (default: 100)

**Процесс:**
1. Создание PipelineExecutor с конфигурацией
2. Запуск выполнения pipeline
3. Возврат результата (количество обработанных, статистика)

**Класс:** `ProcessIssuesBatchUseCase`

---

### 2. ResumeProcessing

**Назначение:** Продолжение обработки после сбоя

**Процесс:**
1. Поиск issues, которые начали обрабатываться, но не завершились
2. Определение этапа, на котором произошел сбой
3. Повторный запуск pipeline для этих issues

**Примечание:** В упрощенной архитектуре просто переобрабатываются все необработанные issues

**Класс:** `ResumeProcessingUseCase`

---

### 3. ReprocessIssue

**Назначение:** Переобработка конкретного issue

**Параметры:**
- `issue_id` - ID issue для переобработки

**Процесс:**
1. Удаление записи из `preprocessed_issues` (если есть)
2. Удаление embedding из ChromaDB (если есть)
3. Запуск pipeline для одного issue

**Класс:** `ReprocessIssueUseCase`

---

### 4. ClusterAllIssues

**Назначение:** Кластеризация всех обработанных issues

**Параметры:**
- `method` - алгоритм (hdbscan/kmeans)
- `use_llm_for_labels` - использовать LLM для названий

**Процесс:**
1. Загрузка всех embeddings из ChromaDB
2. Применение кластеризации
3. Создание кластеров в БД
4. Назначение issues к кластерам
5. Опционально: генерация названий через LLM

**Класс:** `ClusterAllIssuesUseCase`

---

## Задачи разработки

### Phase 1: Инфраструктура и базовая структура

- [ ] Создать структуру проекта согласно DDD
- [ ] Инициализировать uv проект
- [ ] Добавить зависимости:
  ```bash
  uv add fastapi uvicorn psycopg2-binary sqlalchemy
  uv add sentence-transformers chromadb scikit-learn hdbscan
  uv add beautifulsoup4 pymorphy2 pydantic-ai
  uv add --dev pytest pytest-asyncio
  ```
- [ ] Создать миграции для новых таблиц:
  - `preprocessed_issues`
  - `clusters`
  - `message_clusters`

### Phase 2: Domain Layer

#### Domain Models
- [ ] `domain/models/issue.py`
- [ ] `domain/models/message.py`
- [ ] `domain/models/preprocessed_issue.py`
- [ ] `domain/models/embedding.py`
- [ ] `domain/models/cluster.py`

#### Repository Interfaces
- [ ] `domain/repositories/issue_repository.py`
- [ ] `domain/repositories/message_repository.py`
- [ ] `domain/repositories/preprocessed_issue_repository.py`
- [ ] `domain/repositories/cluster_repository.py`

#### Domain Services
- [ ] `domain/services/text_preprocessor.py`
- [ ] `domain/services/embedding_generator.py`
- [ ] `domain/services/vector_db_service.py`
- [ ] `domain/services/clustering_service.py`

### Phase 3: Application Layer

#### Pipeline
- [ ] `application/pipeline/pipeline_config.py`
- [ ] `application/pipeline/pipeline_context.py`
- [ ] `application/pipeline/stages/base_stage.py`
- [ ] `application/pipeline/stages/stage_0_fetch.py`
- [ ] `application/pipeline/stages/stage_1_preprocess.py`
- [ ] `application/pipeline/stages/stage_2_embed.py`
- [ ] `application/pipeline/stages/stage_3_vectordb.py`
- [ ] `application/pipeline/stages/stage_4_complete.py`
- [ ] `application/pipeline/pipeline_executor.py`

#### Use Cases
- [ ] `application/use_cases/process_issues_batch.py`
- [ ] `application/use_cases/resume_processing.py`
- [ ] `application/use_cases/reprocess_issue.py`
- [ ] `application/use_cases/cluster_all_issues.py`

#### DTOs
- [ ] `application/dto/issue_batch_dto.py`
- [ ] `application/dto/processing_result_dto.py`
- [ ] `application/dto/pipeline_stats_dto.py`
- [ ] `application/dto/clustering_result_dto.py`

### Phase 4: Infrastructure Layer

#### Persistence
- [ ] `infrastructure/persistence/models.py` - SQLAlchemy models
- [ ] `infrastructure/persistence/database.py`
- [ ] `infrastructure/persistence/postgres/issue_repository_impl.py`
- [ ] `infrastructure/persistence/postgres/message_repository_impl.py`
- [ ] `infrastructure/persistence/postgres/preprocessed_issue_repository_impl.py`
- [ ] `infrastructure/persistence/postgres/cluster_repository_impl.py`

#### ML Infrastructure
- [ ] `infrastructure/ml/sentence_transformer.py`
- [ ] `infrastructure/ml/text_normalizer.py`
- [ ] `infrastructure/ml/clustering_algorithms.py`

#### Vector DB
- [ ] `infrastructure/vectordb/chromadb_client.py`
- [ ] `infrastructure/vectordb/collection_manager.py`

#### LLM (Optional)
- [ ] `infrastructure/llm/cluster_labeling.py`

### Phase 5: Interface Layer

#### API
- [ ] `interfaces/api/schemas.py`
- [ ] `interfaces/api/routes.py`

### Phase 6: Configuration & Main

- [ ] `config.py`
- [ ] `dependencies.py`
- [ ] `main.py`

### Phase 7: Testing

- [ ] Unit тесты для domain services
- [ ] Unit тесты для pipeline stages
- [ ] Integration тесты для полного pipeline
- [ ] Integration тесты для кластеризации
- [ ] Fixtures для тестовых данных

### Phase 8: Documentation & Docker

- [ ] README.md
- [ ] Docstrings
- [ ] Dockerfile
- [ ] docker-compose.yml
- [ ] .env.example

---

## Критерии готовности

- [ ] Pipeline обрабатывает 100+ issues за раз
- [ ] Промежуточные результаты сохраняются в БД
- [ ] Возможность возобновления после сбоя
- [ ] Embeddings генерируются корректно
- [ ] ChromaDB работает для семантического поиска
- [ ] Кластеризация выполняется отдельной командой
- [ ] Кластеризация группирует схожие issues
- [ ] API endpoints работают
- [ ] Unit тесты проходят (>80% coverage)
- [ ] Integration тесты проходят
- [ ] Документация актуальна

---

## Метрики производительности

**Цель:** Обработка 10000 issues за разумное время

**Benchmarks (примерные):**
- Stage 1 (Preprocessing): ~100-200 issues/sec
- Stage 2 (Embeddings): ~50-100 issues/sec (зависит от GPU)
- Stage 3 (Vector DB): ~500-1000 issues/sec

**Примерная оценка для 10000 issues:**
- Preprocessing: ~50-100 sec
- Embeddings: ~100-200 sec (bottleneck)
- Vector DB: ~10-20 sec
- **Итого: ~3-6 минут**

---

## Конфигурация и переменные окружения

### Основные параметры

**Database:**
- `DATABASE_URL` - подключение к PostgreSQL
- `DB_POOL_SIZE` - размер connection pool

**ChromaDB:**
- `CHROMADB_HOST` - хост ChromaDB
- `CHROMADB_PORT` - порт
- `CHROMADB_COLLECTION` - название коллекции
- `CHROMADB_PERSIST_DIR` - директория персистентного хранилища

**ML Models:**
- `EMBEDDING_MODEL` - модель для embeddings (default: intfloat/multilingual-e5-large)
- `EMBEDDING_DIM` - размерность эмбеддингов
- `MAX_SEQ_LENGTH` - макс. длина последовательности (default: 512)

**Pipeline:**
- `BATCH_SIZE` - размер батча (default: 100)
- `MAX_RETRIES` - количество повторов при ошибке (default: 3)
- `RETRY_DELAY` - задержка между повторами в секундах (default: 5)

**Clustering:**
- `CLUSTERING_METHOD` - алгоритм (hdbscan/kmeans)
- `HDBSCAN_MIN_CLUSTER_SIZE` - минимальный размер кластера (default: 5)
- `HDBSCAN_MIN_SAMPLES` - минимальное количество samples (default: 3)
- `KMEANS_MAX_K` - максимальное K для автоопределения (default: 20)

**LLM (Optional):**
- `LLM_PROVIDER` - провайдер LLM (openai/anthropic/local)
- `LLM_API_KEY` - API ключ
- `LLM_MODEL` - модель для названий кластеров

---

## API Endpoints

### Pipeline Management

**POST /api/v1/analyzer/process**
- Запуск обработки батча issues
- Body: `{ "batch_size": 100 }`
- Response: `ProcessingResult`

**POST /api/v1/analyzer/resume**
- Возобновление обработки после сбоя
- Response: `ProcessingResult`

**POST /api/v1/analyzer/reprocess/{issue_id}**
- Переобработка конкретного issue
- Response: `ProcessingResult`

### Clustering

**POST /api/v1/analyzer/cluster**
- Запуск кластеризации
- Body: `{ "method": "hdbscan", "use_llm": false }`
- Response: `ClusteringResult`

**GET /api/v1/analyzer/clusters**
- Список всех кластеров
- Query params: `limit`, `offset`, `sort_by`
- Response: `List[Cluster]`

**GET /api/v1/analyzer/clusters/{cluster_id}**
- Детали кластера
- Response: `ClusterDetail` (с issues)

### Search

**POST /api/v1/analyzer/search/similar**
- Семантический поиск похожих issues
- Body: `{ "issue_id": "uuid", "top_k": 10 }`
- Response: `List[SimilarIssue]`

**POST /api/v1/analyzer/search/text**
- Поиск по тексту запроса
- Body: `{ "query": "text", "top_k": 10 }`
- Response: `List[SimilarIssue]`

### Statistics

**GET /api/v1/analyzer/stats**
- Общая статистика обработки
- Response: статистика (total_issues, processed, pending, clusters_count)

---

## Зависимости между сервисами

### Database Service
- Analyzer зависит от БД для чтения issues и сохранения preprocessed_issues
- Использует существующие таблицы: issues, messages
- Создает новые таблицы: preprocessed_issues, clusters, message_clusters

### Parser Service
- Analyzer читает данные, которые Parser импортировал
- Нет прямого взаимодействия между сервисами

### Query Service (будущее)
- Query Service будет использовать результаты Analyzer
- Семантический поиск через ChromaDB
- Фильтрация по кластерам

---

## Расширение функционала (будущее)

### Анализ messages
- Сейчас: обрабатываем только issue.title + issue.description
- Будущее: анализ всех messages в issue для более глубокого понимания

### Intent extraction
- Сейчас: только кластеризация
- Будущее: определение намерений пользователей через LLM
- Создание и заполнение таблицы `intents`

### Real-time processing
- Сейчас: batch обработка
- Будущее: обработка новых issues в реальном времени (event-driven)

### Advanced clustering
- Иерархическая кластеризация
- Динамическое обновление кластеров при добавлении новых issues
- Связывание кластеров между собой

---

## Примеры использования

### CLI команды

```bash
# Обработать 100 необработанных issues
python -m analyzer process --batch-size 100

# Переобработать конкретный issue
python -m analyzer reprocess <issue-id>

# Запустить кластеризацию
python -m analyzer cluster --method hdbscan --use-llm

# Посмотреть статистику
python -m analyzer stats

# Найти похожие issues
python -m analyzer search --issue-id <id> --top-k 10
```

### API примеры

```bash
# Запуск обработки
curl -X POST http://localhost:8000/api/v1/analyzer/process \
  -H "Content-Type: application/json" \
  -d '{"batch_size": 100}'

# Кластеризация
curl -X POST http://localhost:8000/api/v1/analyzer/cluster \
  -H "Content-Type: application/json" \
  -d '{"method": "hdbscan", "use_llm": true}'

# Семантический поиск
curl -X POST http://localhost:8000/api/v1/analyzer/search/similar \
  -H "Content-Type: application/json" \
  -d '{"issue_id": "uuid-here", "top_k": 10}'
```

---

## Troubleshooting

### Частые проблемы

**Pipeline падает на Stage 2:**
- Проверить доступность модели embeddings
- Проверить наличие GPU/CPU ресурсов
- Уменьшить batch_size

**ChromaDB connection failed:**
- Проверить CHROMADB_HOST и CHROMADB_PORT
- Убедиться, что ChromaDB запущен
- Проверить права доступа к CHROMADB_PERSIST_DIR

**Медленная обработка:**
- Увеличить batch_size для embeddings
- Использовать GPU вместо CPU
- Выбрать более легкую модель (rubert-tiny2)

**Кластеризация создает слишком много/мало кластеров:**
- Настроить HDBSCAN_MIN_CLUSTER_SIZE
- Попробовать K-means с фиксированным K
- Увеличить количество обработанных issues перед кластеризацией

---

## Безопасность

### Рекомендации

1. **Изоляция данных:**
   - ChromaDB в отдельном контейнере
   - Ограничение доступа к БД только для сервиса

2. **Валидация входных данных:**
   - Проверка batch_size (макс. 1000)
   - Валидация issue_id перед обработкой

3. **Rate limiting:**
   - Ограничение запросов к API
   - Защита от DoS на endpoints обработки

4. **Логирование:**
   - Логирование всех операций pipeline
   - Аудит доступа к sensitive данным

---
