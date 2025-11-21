# Архитектура системы
## Система анализа намерений пользователей службы поддержки

**Версия:** 1.0
**Дата:** 21.11.2025

---

## 1. Обзор архитектуры

Система построена на микросервисной архитектуре с использованием Docker-контейнеров для локального использования одним пользователем. Все сервисы изолированы и взаимодействуют через API и общую базу данных PostgreSQL.

### 1.1 Принципы архитектуры
- **Микросервисная архитектура** - независимые сервисы в отдельных контейнерах
- **Изоляция** - каждый сервис работает в собственном Docker-контейнере
- **Гибридное хранилище** - PostgreSQL для структурированных данных + ChromaDB для векторов
- **Локальное развертывание** - система работает на одной машине
- **Pipeline обработка** - многоэтапная обработка issues (preprocessing → embeddings → vector DB)
- **Issue-centric approach** - обработка целых issues (title + description), не отдельных messages
- **Отложенная кластеризация** - кластеризация выполняется отдельно после обработки
- **Семантический поиск** - использование векторных представлений для поиска похожих issues

---

## 2. Компоненты системы

### 2.1 Диаграмма компонентов

```
┌─────────────────────────────────────────────────────────┐
│                    Пользователь                          │
└────────────┬──────────────────────────┬─────────────────┘
             │                          │
             │ HTTP                     │ CLI
             ▼                          ▼
    ┌────────────────┐        ┌──────────────────┐
    │   GUI Service  │        │   CLI Service    │
    │   (Web UI)     │        │   (Commands)     │
    └────────┬───────┘        └────────┬─────────┘
             │                          │
             │ REST API                 │ Direct calls
             ▼                          ▼
    ┌─────────────────────────────────────────────┐
    │           API Gateway Service                │
    │       (Маршрутизация запросов)              │
    └────────┬──────────┬──────────┬───────────────┘
             │          │          │
             ▼          ▼          ▼
    ┌────────────┐ ┌──────────┐ ┌─────────────────┐
    │  Parser    │ │ Analyzer │ │  Query Service  │
    │  Service   │ │ Service  │ │  (Поиск/Отчеты) │
    └─────┬──────┘ └────┬─────┘ └────────┬────────┘
          │             │ │                │
          │             │ └────────┐       │
          └─────────────┴──────────┼───────┘
                        │          │
                        ▼          ▼
             ┌──────────────────────┐  ┌─────────────┐
             │   PostgreSQL DB      │  │  ChromaDB   │
             │ (Структурные данные) │  │ (Векторы)   │
             └──────────────────────┘  └─────────────┘
```

---

## 3. Описание сервисов

### 3.1 Database Service (support-db)
**Назначение:** Централизованное хранилище данных

**Технологии:**
- PostgreSQL 13+
- Docker volume для персистентности

**Схема данных:**
```sql
-- Enums
CREATE TYPE source_type AS ENUM ('okdesk', 'telegram');
CREATE TYPE issue_status AS ENUM ('opened', 'wait', 'completed', 'closed');
CREATE TYPE author_type AS ENUM ('employee', 'contact', 'user');
CREATE TYPE import_status AS ENUM ('in_progress', 'completed', 'failed');
CREATE TYPE tag_type AS ENUM ('auto', 'okdesk', 'manual');

-- Источники данных
CREATE TABLE sources (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  type source_type NOT NULL,
  config JSONB,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Обращения
CREATE TABLE issues (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  external_id VARCHAR(255) NOT NULL,
  source_id UUID REFERENCES sources(id),
  title TEXT,
  description TEXT,
  status issue_status,
  priority INTEGER CHECK (priority BETWEEN 1 AND 4),
  created_at TIMESTAMP,
  updated_at TIMESTAMP,
  completed_at TIMESTAMP,
  UNIQUE(external_id, source_id)
);

-- Сообщения
CREATE TABLE messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  issue_id UUID REFERENCES issues(id),
  external_id VARCHAR(255) NOT NULL,
  author_id VARCHAR(255),
  author_name TEXT,
  author_type author_type,
  content TEXT NOT NULL,
  is_public BOOLEAN DEFAULT true,
  published_at TIMESTAMP,
  UNIQUE(external_id, issue_id)
);

-- Намерения (динамический каталог)
CREATE TABLE intents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  code TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  description TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Результаты анализа
CREATE TABLE message_analysis (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  message_id UUID REFERENCES messages(id) UNIQUE,
  analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  reasoning TEXT
);

-- Связь сообщений с намерениями
CREATE TABLE message_intents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  message_analysis_id UUID REFERENCES message_analysis(id),
  intent_id UUID REFERENCES intents(id),
  confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
  UNIQUE(message_analysis_id, intent_id)
);

-- Теги
CREATE TABLE tags (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL UNIQUE,
  type tag_type,
  source VARCHAR(50),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Связь сообщений и тегов
CREATE TABLE message_tags (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  message_analysis_id UUID REFERENCES message_analysis(id),
  tag_id UUID REFERENCES tags(id),
  confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
  UNIQUE(message_analysis_id, tag_id)
);

-- Предобработанные issues (результат Stage 1)
CREATE TABLE preprocessed_issues (
  id UUID PRIMARY KEY REFERENCES issues(id),
  content TEXT NOT NULL,  -- preprocessed title + description
  processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Кластеры issues (результат кластеризации)
CREATE TABLE clusters (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  cluster_label INT NOT NULL UNIQUE,  -- метка из алгоритма
  name TEXT,  -- название (опционально через LLM)
  description TEXT,  -- описание (опционально через LLM)
  centroid_embedding vector(1024),  -- центроид кластера
  size INT DEFAULT 0,  -- количество issues
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Связь issues с кластерами
CREATE TABLE message_clusters (
  message_id UUID REFERENCES messages(id),
  cluster_id UUID REFERENCES clusters(id),
  distance_to_centroid FLOAT,  -- расстояние до центроида (0-1)
  assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (message_id, cluster_id)
);

-- История импортов
CREATE TABLE imports (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  source_id UUID REFERENCES sources(id),
  filename TEXT,
  file_path TEXT,
  started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  completed_at TIMESTAMP,
  status import_status,
  stats JSONB,
  error_message TEXT
);

-- Индексы
CREATE INDEX idx_messages_issue_id ON messages(issue_id);
CREATE INDEX idx_messages_published_at ON messages(published_at);
CREATE INDEX idx_message_analysis_message_id ON message_analysis(message_id);
CREATE INDEX idx_issues_external_id ON issues(external_id);
CREATE INDEX idx_issues_created_at ON issues(created_at);
CREATE INDEX idx_preprocessed_issues_processed ON preprocessed_issues(processed_at);
CREATE INDEX idx_preprocessed_issues_text ON preprocessed_issues USING gin(to_tsvector('russian', content));
CREATE INDEX idx_clusters_size ON clusters(size DESC);
CREATE INDEX idx_message_clusters_cluster_id ON message_clusters(cluster_id);
CREATE INDEX idx_message_clusters_distance ON message_clusters(distance_to_centroid);
CREATE INDEX idx_message_intents_intent_id ON message_intents(intent_id);
CREATE INDEX idx_message_tags_tag_id ON message_tags(tag_id);
CREATE INDEX idx_intents_code ON intents(code);
```

**Примечание:** Таблица `intents` начинается пустой и заполняется автоматически при анализе сообщений LLM агентом.

### 3.2 Parser Service
**Назначение:** Парсинг и импорт данных из различных источников

**Технологии:**
- Python 3.12
- FastAPI для API
- BeautifulSoup4 для HTML парсинга
- psycopg2 для работы с PostgreSQL
- json для парсинга

**Функции:**
- Чтение файлов JSONL (OKDesk) и JSON (Telegram)
- Извлечение сообщений и метаданных напрямую из JSON
- Дедупликация по external_id
- Сохранение в БД с транзакциями
- Автоматический вызов анализа после импорта

**API endpoints:**
- `POST /api/v1/import/okdesk` - загрузка файла OKDesk
- `POST /api/v1/import/telegram` - загрузка файла Telegram
- `GET /api/v1/import/{id}` - статус импорта
- `GET /api/v1/imports` - история импортов

**Рабочий процесс импорта:**
1. Получение файла JSONL
2. Создание записи в таблице `imports` со статусом 'in_progress'
3. Построчное чтение файла, парсинг JSON напрямую
4. Для каждой строки:
   - Извлечение issue и comments из JSON
   - Проверка дубликата
   - INSERT/UPDATE в `issues` и `messages`
5. Обновление статуса импорта на 'completed'
6. Вызов Analyzer Service с пачкой message_ids
7. Возврат статистики

### 3.3 Analyzer Service
**Назначение:** Pipeline-обработка issues с генерацией эмбеддингов и кластеризацией

**Технологии:**
- Python 3.12
- FastAPI для API
- **SentenceTransformers** - для генерации эмбеддингов (multilingual-e5-large)
- **ChromaDB** - векторное хранилище для семантического поиска
- **HDBSCAN / K-means** - алгоритмы кластеризации
- BeautifulSoup4 для очистки HTML
- pymorphy2 для лемматизации
- psycopg2 для работы с PostgreSQL

**Архитектура:** DDD (Domain-Driven Design) + Pipeline Processing

**Pipeline обработки (4 этапа):**

**Stage 0: Data Fetching**
- Получение N необработанных issues из БД
- SQL: `SELECT * FROM issues WHERE id NOT IN (SELECT id FROM preprocessed_issues)`
- Загрузка связанных messages (для будущего расширения)

**Stage 1: Preprocessing**
- Извлечение issue.title и issue.description
- Очистка HTML/XML тегов (BeautifulSoup)
- Нормализация текста (lower, spaces, punctuation)
- Удаление стоп-слов и шаблонных фраз
- Лемматизация (pymorphy2)
- Токенизация спец. паттернов ([URL], [EMAIL], [PHONE])
- Объединение: `"{title_processed}\n\n{description_processed}"`
- Сохранение в `preprocessed_issues` (id, content, processed_at)

**Stage 2: Embedding Generation**
- Генерация векторных представлений через SentenceTransformer
- Batch обработка (32 issue за раз)
- L2 нормализация векторов
- Embeddings передаются в Stage 3 (не сохраняются в PostgreSQL)

**Stage 3: Vector DB Storage**
- Сохранение embeddings в ChromaDB
- Идентификатор: issue.id (UUID)
- Document: preprocessed content
- Метрика: cosine similarity
- Связь с PostgreSQL только через issue.id

**Stage 4: Completion**
- Логирование успешной обработки
- Обновление статистики

**Кластеризация (отдельная команда):**
- Выполняется после обработки всех/большинства issues
- Загрузка всех embeddings из ChromaDB
- Применение HDBSCAN (автоопределение кластеров) или K-means
- Вычисление центроидов и расстояний
- Сохранение в таблицы: `clusters`, `message_clusters`
- Опционально: генерация названий через LLM

**API endpoints:**
- `POST /api/v1/analyzer/process` - запуск pipeline обработки
  - Body: `{batch_size: 100}`
  - Response: `{processed_count, duration, stats}`
- `POST /api/v1/analyzer/resume` - возобновление после сбоя
- `POST /api/v1/analyzer/reprocess/{issue_id}` - переобработка issue
- `POST /api/v1/analyzer/cluster` - запуск кластеризации
  - Body: `{method: "hdbscan|kmeans", use_llm: false}`
- `POST /api/v1/analyzer/search/similar` - семантический поиск
  - Body: `{issue_id: "uuid", top_k: 10}`
- `GET /api/v1/analyzer/stats` - статистика обработки

**Производительность:**
- Preprocessing: ~100-200 issues/sec
- Embeddings: ~50-100 issues/sec (bottleneck, зависит от GPU)
- Vector DB: ~500-1000 issues/sec
- **Итого: 10000 issues за ~3-6 минут**

**Новые таблицы БД:**
- `preprocessed_issues` - предобработанный текст issues
- `clusters` - кластеры схожих issues
- `message_clusters` - связь issues с кластерами

### 3.4 Query Service
**Назначение:** Поиск, фильтрация и генерация отчетов

**Технологии:**
- Python 3.12
- FastAPI для API
- SQLAlchemy 2.0 для ORM
- ChromaDB client для семантического поиска
- psycopg2

**Функции:**
- **Семантический поиск** - поиск похожих issues через векторные представления
- Полнотекстовый поиск по сообщениям (PostgreSQL full-text)
- Фильтрация по кластерам, тегам, датам, источникам
- Просмотр кластеров и их содержимого
- Агрегация статистики
- Генерация отчетов (JSON, CSV)

**API endpoints:**
- `POST /api/v1/search/semantic` - семантический поиск похожих issues
  - Body: `{query: "текст" | issue_id: "uuid", top_k: 10}`
- `GET /api/v1/search` - полнотекстовый поиск
- `GET /api/v1/issues` - список обращений с фильтрами
- `GET /api/v1/issues/{id}` - детали обращения
- `GET /api/v1/clusters` - список кластеров
- `GET /api/v1/clusters/{id}` - детали кластера с issues
- `GET /api/v1/stats/processing` - статистика обработки
- `GET /api/v1/stats/clusters` - статистика по кластерам
- `GET /api/v1/stats/sources` - статистика по источникам
- `GET /api/v1/stats/timeline` - временная статистика
- `POST /api/v1/export` - экспорт данных

### 3.5 API Gateway Service
**Назначение:** Единая точка входа и маршрутизация

**Технологии:**
- Python 3.12
- FastAPI
- httpx для HTTP запросов

**Функции:**
- Маршрутизация к соответствующим сервисам
- Логирование запросов
- CORS middleware

**Структура маршрутизации:**
```
/api/v1/import/*   -> Parser Service
/api/v1/analyze/*  -> Analyzer Service
/api/v1/search     -> Query Service
/api/v1/issues/*   -> Query Service
/api/v1/stats/*    -> Query Service
/api/v1/clusters/* -> Query Service
/api/v1/export     -> Query Service
/api/v1/health     -> Health checks
```

### 3.6 CLI Service
**Назначение:** Интерфейс командной строки

**Технологии:**
- Python 3.12
- Click или Typer
- Rich для форматированного вывода
- httpx для HTTP запросов

**Примеры команд:**
```bash
# Импорт
support-cli import okdesk /data/okdesk/out.jsonl
support-cli import telegram /data/telegram/result.json

# Анализ
support-cli analyze all --batch-size 20
support-cli analyze recent 100

# Поиск
support-cli search "проблема с оплатой"
support-cli search intent "нужен_единорог"

# Статистика
support-cli stats intents
support-cli stats tags
support-cli stats timeline --from 2025-01-01

# Экспорт
support-cli export csv --intent проблема --output /tmp/problems.csv
```

### 3.7 GUI Service
**Назначение:** Веб-интерфейс

**Технологии:**
- React 18+ (TypeScript)
- Material-UI или Ant Design
- Chart.js или Recharts
- Axios
- React Router

**Страницы:**
- **Dashboard** - общая статистика, графики
- **Импорт** - загрузка файлов, история импортов
- **Обращения** - список с фильтрами, поиск
- **Детали обращения** - сообщения с намерениями и тегами
- **Аналитика** - графики по намерениям, тегам, временная динамика
- **Кластеры** - группировка сообщений
- **Поиск** - полнотекстовый поиск с фильтрами
- **Экспорт** - генерация отчетов

---

## 4. Потоки данных

### 4.1 Импорт и пакетный анализ

```
┌──────────┐     ┌─────────┐     ┌──────────┐     ┌─────────┐
│  Файл    │────▶│ Parser  │────▶│   DB     │────▶│Analyzer │
│ OKDesk   │     │ Service │     │ Messages │     │Service  │
└──────────┘     └─────────┘     └──────────┘     └────┬────┘
                                                        │
                                                        │ Batch (10-50)
                                                        ▼
                                                 ┌────────────────┐
                                                 │  pydantic_ai   │
                                                 │  LLM Agent     │
                                                 └───────┬────────┘
                                                         │
                                                         │ Results
                                                         ▼
                                                  ┌─────────────────┐
                                                  │ DB (intents,    │
                                                  │ tags, analysis) │
                                                  └─────────────────┘
```

**Последовательность:**
1. Parser читает файл JSONL построчно
2. Для каждой строки извлекает issue и comments
3. Сохраняет в БД (issues, messages)
4. После завершения импорта собирает список новых message_ids
5. Вызывает Analyzer с пачкой IDs
6. Analyzer загружает сообщения, очищает HTML
7. Отправляет батч (10-50 сообщений) в pydantic_ai агента
8. Агент анализирует и возвращает намерения/теги для каждого
9. Analyzer сохраняет результаты в БД
10. Динамически создаются новые намерения при необходимости

---

## 5. Развертывание

### 5.1 Docker Compose

```yaml
version: '3.8'

services:
  db:
    build: ./db
    container_name: support-db
    volumes:
      - pg_data:/var/lib/postgresql/data
      - ./data:/opt/support/data:ro
    ports:
      - "15432:5432"
    environment:
      POSTGRES_HOST_AUTH_METHOD: trust
    networks:
      - support-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  parser:
    build: ./services/parser
    container_name: support-parser
    depends_on:
      db:
        condition: service_healthy
    volumes:
      - ./data:/app/data:ro
    environment:
      DB_HOST: db
      ANALYZER_URL: http://analyzer:8002
    ports:
      - "8001:8001"
    networks:
      - support-network

  analyzer:
    build: ./services/analyzer
    container_name: support-analyzer
    depends_on:
      db:
        condition: service_healthy
    environment:
      DB_HOST: db
      LLM_PROVIDER: openai
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      BATCH_SIZE: 10
    ports:
      - "8002:8002"
    networks:
      - support-network

  query:
    build: ./services/query
    container_name: support-query
    depends_on:
      db:
        condition: service_healthy
    environment:
      DB_HOST: db
    ports:
      - "8003:8003"
    networks:
      - support-network

  api-gateway:
    build: ./services/api-gateway
    container_name: support-api-gateway
    depends_on:
      - parser
      - analyzer
      - query
    environment:
      PARSER_URL: http://parser:8001
      ANALYZER_URL: http://analyzer:8002
      QUERY_URL: http://query:8003
    ports:
      - "8000:8000"
    networks:
      - support-network

  gui:
    build: ./services/gui
    container_name: support-gui
    depends_on:
      - api-gateway
    environment:
      REACT_APP_API_URL: http://localhost:8000
    ports:
      - "3000:3000"
    networks:
      - support-network

volumes:
  pg_data:
    name: support-db-pg-data

networks:
  support-network:
    driver: bridge
```

### 5.2 Структура проекта

```
okdesk/
├── docker-compose.yml
├── .env.example
├── .gitignore
├── README.md
├── docs/
│   ├── PRD.md
│   └── ARCHITECTURE.md
├── db/
│   ├── Dockerfile
│   ├── Makefile
│   ├── init/
│   └── migrations/
├── data/
│   ├── okdesk/
│   └── telegram/
└── services/
    ├── parser/
    ├── analyzer/
    ├── query/
    ├── api-gateway/
    ├── cli/
    └── gui/
```

---

## 6. Технологический стек

### 6.1 Backend
- **Язык:** Python 3.12
- **Web Framework:** FastAPI
- **ORM:** SQLAlchemy 2.0
- **Database Driver:** psycopg2-binary
- **CLI:** Click или Typer
- **HTTP Client:** httpx
- **HTML Parsing:** BeautifulSoup4
- **LLM Agents:** **pydantic_ai**
- **ML/Clustering:** scikit-learn
- **Logging:** Python logging + Rich

### 6.2 Frontend
- **Framework:** React 18+ (TypeScript)
- **UI Library:** Material-UI или Ant Design
- **Charts:** Chart.js или Recharts
- **HTTP:** Axios
- **Routing:** React Router 6

### 6.3 Database
- **СУБД:** PostgreSQL 13+
- **Расширения:** pg_trgm для полнотекстового поиска

### 6.4 Infrastructure
- **Контейнеризация:** Docker, Docker Compose

---

## 7. Безопасность

- Изоляция сервисов в Docker контейнерах
- Внутренняя Docker сеть для межсервисного взаимодействия
- Только API Gateway и GUI доступны на localhost извне
- Роли БД с ограниченными правами (support, support_admin)
- LLM API ключи через environment variables
- ORM защита от SQL injection
- Санитизация HTML через BeautifulSoup

---

## 8. Логирование и мониторинг

- Структурированные логи в stdout/stderr
- Health check endpoints для каждого сервиса
- Сбор логов через `docker-compose logs`

---

## 9. Ограничения и компромиссы

- **Локальное использование** - один пользователь
- **Синхронная обработка** - без очередей
- **Пакетная обработка** - batch размер 10-50 сообщений
- **Зависимость от LLM** - через pydantic_ai
- **Динамическое создание намерений** - через LLM
- **UUID вместо AUTO_INCREMENT** - для гибкости
- **Множественные намерения и теги** - через связующие таблицы
- **Использование ENUM** - для статических типов (status, author_type, etc.)
