# Analyzer Service - Декомпозиция задач

**Сервис:** support-analyzer
**Приоритет:** Критический (основной функционал)
**Технологии:** Python 3.12, FastAPI, pydantic-ai, scikit-learn, uv
**Архитектура:** DDD (Domain-Driven Design)

---

## Обзор

Analyzer Service выполняет пакетный анализ сообщений с помощью LLM агентов (pydantic-ai). Динамически определяет намерения, создает теги и группирует схожие сообщения в кластеры.

---

## DDD Структура

```
services/analyzer/
├── src/
│   ├── domain/              # Domain layer
│   │   ├── models/
│   │   │   ├── message.py
│   │   │   ├── intent.py
│   │   │   ├── tag.py
│   │   │   ├── analysis.py
│   │   │   └── cluster.py
│   │   ├── repositories/
│   │   │   ├── message_repository.py
│   │   │   ├── intent_repository.py
│   │   │   ├── tag_repository.py
│   │   │   ├── analysis_repository.py
│   │   │   └── cluster_repository.py
│   │   └── services/
│   │       ├── text_processor.py
│   │       └── clustering_service.py
│   ├── application/         # Application layer
│   │   ├── use_cases/
│   │   │   ├── analyze_batch.py
│   │   │   ├── analyze_all.py
│   │   │   └── reanalyze_message.py
│   │   └── dto/
│   │       ├── analysis_dto.py
│   │       └── batch_dto.py
│   ├── infrastructure/      # Infrastructure layer
│   │   ├── persistence/
│   │   │   ├── postgres/
│   │   │   │   ├── message_repository_impl.py
│   │   │   │   ├── intent_repository_impl.py
│   │   │   │   ├── tag_repository_impl.py
│   │   │   │   ├── analysis_repository_impl.py
│   │   │   │   └── cluster_repository_impl.py
│   │   │   ├── models.py
│   │   │   └── database.py
│   │   ├── llm/
│   │   │   ├── pydantic_ai_agent.py
│   │   │   └── models.py          # Pydantic models для LLM
│   │   └── ml/
│   │       └── clustering.py       # scikit-learn clustering
│   ├── interfaces/          # Interface adapters
│   │   └── api/
│   │       ├── routes.py
│   │       └── schemas.py
│   ├── config.py
│   ├── dependencies.py
│   └── main.py
├── tests/
│   ├── unit/
│   │   ├── domain/
│   │   ├── application/
│   │   └── infrastructure/
│   ├── integration/
│   └── fixtures/
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Задачи

### 1. Настройка проекта

- [ ] Создать структуру директорий согласно DDD
- [ ] Инициализировать uv проект
- [ ] Добавить зависимости
  ```bash
  uv add fastapi uvicorn psycopg2-binary sqlalchemy
  uv add beautifulsoup4 pydantic-ai scikit-learn
  uv add --dev pytest pytest-asyncio
  ```
- [ ] Создать pyproject.toml с настройками
- [ ] Создать Dockerfile (multi-stage с uv)

### 2. Domain Layer

#### 2.1 Domain Models

- [ ] Создать `domain/models/message.py`
  - Dataclass Message
  - Поля: id, issue_id, content, author_info, published_at

- [ ] Создать `domain/models/intent.py`
  - Dataclass Intent
  - Поля: id, code, name, description
  - Value Object IntentWithConfidence (intent, confidence)

- [ ] Создать `domain/models/tag.py`
  - Dataclass Tag
  - Enum TagType
  - Value Object TagWithConfidence

- [ ] Создать `domain/models/analysis.py`
  - Dataclass MessageAnalysis
  - Поля: id, message_id, intents (list), tags (list), reasoning, analyzed_at

- [ ] Создать `domain/models/cluster.py`
  - Dataclass IntentCluster
  - Dataclass MessageCluster

#### 2.2 Repository Interfaces

- [ ] Создать `domain/repositories/message_repository.py`
  - Методы: get_by_id, get_batch, get_unanalyzed

- [ ] Создать `domain/repositories/intent_repository.py`
  - Методы: get_by_code, create, get_all

- [ ] Создать `domain/repositories/tag_repository.py`
  - Методы: get_by_name, create, bulk_get_or_create

- [ ] Создать `domain/repositories/analysis_repository.py`
  - Методы: create, get_by_message_id, exists

- [ ] Создать `domain/repositories/cluster_repository.py`
  - Методы: get_all, create, add_message_to_cluster

#### 2.3 Domain Services

- [ ] Создать `domain/services/text_processor.py`
  - Класс TextProcessor
  - Методы:
    - clean_html(text) -> str (BeautifulSoup)
    - normalize_text(text) -> str (пробелы, переносы)
    - extract_clean_text(html_content) -> str

- [ ] Создать `domain/services/clustering_service.py`
  - Класс ClusteringService
  - Методы:
    - find_similar_messages(message_vectors) -> clusters
    - assign_to_cluster(message_vector, existing_clusters) -> cluster_id

### 3. Application Layer

#### 3.1 DTOs

- [ ] Создать `application/dto/analysis_dto.py`
  - AnalysisRequest
  - AnalysisResult
  - IntentResult (code, name, description, confidence)
  - MessageAnalysisResult (message_id, intents, tags, reasoning)

- [ ] Создать `application/dto/batch_dto.py`
  - BatchRequest (message_ids, batch_size)
  - BatchResponse (total, processed, failed, results)
  - BatchAnalysisResult (messages, common_intents)

#### 3.2 Use Cases

- [ ] Создать `application/use_cases/analyze_batch.py`
  - Класс AnalyzeBatchUseCase
  - Зависимости: repositories, text_processor, llm_agent, clustering_service
  - Метод execute(message_ids, batch_size) -> BatchResponse
  - Алгоритм:
    1. Разбить message_ids на батчи
    2. Для каждого батча:
       - Загрузить messages из БД
       - Очистить HTML
       - Вызвать pydantic-ai агента
       - Обработать результаты
       - Сохранить analysis, intents, tags
       - Выполнить кластеризацию (опционально)
    3. Вернуть статистику

- [ ] Создать `application/use_cases/analyze_all.py`
  - Класс AnalyzeAllUseCase
  - Получить все unanalyzed messages
  - Вызвать AnalyzeBatchUseCase

- [ ] Создать `application/use_cases/reanalyze_message.py`
  - Класс ReanalyzeMessageUseCase
  - Удалить старый analysis
  - Выполнить новый анализ

### 4. Infrastructure Layer

#### 4.1 Persistence

- [ ] Создать `infrastructure/persistence/models.py`
  - SQLAlchemy модели:
    - MessageModel
    - IntentModel
    - TagModel
    - MessageAnalysisModel
    - MessageIntentModel
    - MessageTagModel
    - IntentClusterModel
    - MessageClusterModel

- [ ] Создать `infrastructure/persistence/database.py`
  - Engine и SessionLocal

- [ ] Реализовать все repository implementations в `persistence/postgres/`
  - MessageRepositoryImpl
  - IntentRepositoryImpl
  - TagRepositoryImpl
  - AnalysisRepositoryImpl
  - ClusterRepositoryImpl

#### 4.2 LLM Integration (pydantic-ai)

- [ ] Создать `infrastructure/llm/models.py`
  - Pydantic модели для структурированного вывода:
    - IntentResult (code, name, description, confidence)
    - MessageAnalysisResult (message_id, intents, tags, reasoning)
    - BatchAnalysisResult (messages, common_intents)

- [ ] Создать `infrastructure/llm/pydantic_ai_agent.py`
  - Класс PydanticAIAnalyzer
  - Инициализация Agent из pydantic-ai:
    - model = 'openai:gpt-4' или 'anthropic:claude-3-5-sonnet'
    - result_type = BatchAnalysisResult
    - system_prompt с инструкциями
  - Метод analyze_batch(messages) -> BatchAnalysisResult
    - Формирование промпта с текстами сообщений
    - Вызов агента
    - Возврат структурированного результата

- [ ] Создать system_prompt для агента
  - Описание задачи: анализ пачки сообщений
  - Инструкции по намерениям (динамические, snake_case)
  - Требование консистентности кодов в пачке
  - Формат ответа через Pydantic модели
  - Примеры намерений: запрос_информации, нужен_единорог, etc

#### 4.3 ML (Clustering)

- [ ] Создать `infrastructure/ml/clustering.py`
  - Класс MLClusteringService (реализация ClusteringService)
  - Методы:
    - vectorize_text(text) -> vector (TF-IDF или embeddings)
    - cluster_messages(vectors) -> cluster_labels
    - calculate_similarity(vector1, vector2) -> float
  - Использование scikit-learn:
    - TfidfVectorizer
    - KMeans или DBSCAN
    - cosine_similarity

### 5. Interface Layer (API)

#### 5.1 Schemas

- [ ] Создать `interfaces/api/schemas.py`
  - BatchAnalysisRequest
  - BatchAnalysisResponse
  - MessageAnalysisResponse
  - ErrorResponse

#### 5.2 Routes

- [ ] Создать `interfaces/api/routes.py`
  - POST /api/v1/analyze/batch
    - Body: {message_ids, batch_size}
    - Вызов AnalyzeBatchUseCase
    - Response: статистика обработки
  - POST /api/v1/analyze/all
    - Вызов AnalyzeAllUseCase
  - POST /api/v1/analyze/message/{id}
    - Анализ одного сообщения (для тестирования)
  - POST /api/v1/reanalyze/{id}
    - Повторный анализ

### 6. Configuration and DI

- [ ] Создать `config.py`
  - Settings:
    - DB connection
    - LLM provider (openai/anthropic)
    - API keys
    - Batch size (default 10)
    - Clustering settings

- [ ] Создать `dependencies.py`
  - Функции для DI:
    - get_db()
    - get_message_repository()
    - get_intent_repository()
    - get_tag_repository()
    - get_analysis_repository()
    - get_cluster_repository()
    - get_text_processor()
    - get_llm_agent()
    - get_clustering_service()
    - get_analyze_batch_use_case()

- [ ] Создать `main.py`
  - FastAPI app
  - Include routers
  - Health check
  - Exception handlers

### 7. Тестирование

- [ ] Unit тесты для domain services
  - tests/unit/domain/test_text_processor.py
  - tests/unit/domain/test_clustering_service.py

- [ ] Unit тесты для use cases (с моками)
  - tests/unit/application/test_analyze_batch.py

- [ ] Интеграционные тесты
  - tests/integration/test_analyze_batch.py
  - Использовать реальный LLM (или мок)
  - Проверить сохранение в БД

- [ ] Тесты для LLM агента
  - tests/unit/infrastructure/test_pydantic_ai_agent.py
  - Мокировать вызовы LLM

### 8. Документация

- [ ] Создать README.md
  - Описание сервиса
  - Архитектура DDD
  - Описание пакетного анализа
  - Настройка LLM провайдера
  - API документация

- [ ] Docstrings для всех классов и методов

---

## Критерии готовности

- [ ] Все слои DDD реализованы
- [ ] pydantic-ai агент работает
- [ ] Пакетный анализ работает
- [ ] Динамическое создание намерений работает
- [ ] Сохранение результатов в БД корректно
- [ ] Кластеризация работает (опционально)
- [ ] Unit тесты проходят
- [ ] Интеграционные тесты проходят
- [ ] Документация актуальна

---

## Зависимости

**Требует:**
- Database Service готов
- Parser Service (для вызова после импорта)

**Блокирует:**
- Query Service (нужны данные анализа)
- GUI (отображение результатов)

**Интегрируется с:**
- Parser Service вызывает Analyzer после импорта
