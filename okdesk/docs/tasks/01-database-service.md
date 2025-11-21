# Database Service - Декомпозиция задач

**Сервис:** support-db
**Приоритет:** Критический (блокирует все остальные сервисы)
**Технологии:** PostgreSQL 13+, Docker, dbmate

---

## Обзор

Database Service - это централизованное хранилище данных для всей системы. Включает схему БД, миграции, роли и права доступа.

---

## Задачи

### 1. Настройка окружения

- [ ] Создать структуру директории `db/`
  ```
  db/
  ├── Dockerfile
  ├── Makefile
  ├── VERSION
  ├── .env
  ├── .gitignore
  ├── init/
  │   ├── 01-create-role.sql
  │   ├── 02-tablespace.sql
  │   └── 03-database.sql
  ├── migrations/
  │   └── schema.sql
  └── changelog.d/
  ```

- [ ] Создать Dockerfile
  - Базовый образ: postgres:13
  - Копирование init скриптов
  - Настройка локали UTF-8

- [ ] Создать Makefile
  - Команды: build, run, stop, migrate, new-migration
  - Переменные: DB_PORT, VOLUMES, ENV

- [ ] Создать .env файл
  - DATABASE_URL для dbmate
  - Параметры подключения

### 2. Роли и права доступа

- [ ] Создать `init/01-create-role.sql`
  - Роль `support` - для приложений (LOGIN, NOCREATEDB, NOCREATEROLE)
  - Роль `support_admin` - для миграций (SUPERUSER)
  - Пароли из env

- [ ] Создать `init/02-tablespace.sql` (если нужен отдельный tablespace)

- [ ] Создать `init/03-database.sql`
  - Создание БД `support`
  - Владелец: support_admin
  - Encoding: UTF8

### 3. Миграции - ENUM типы

- [ ] Создать миграцию `YYYYMMDD_enum_types.sql`
  ```sql
  CREATE TYPE source_type AS ENUM ('okdesk', 'telegram');
  CREATE TYPE issue_status AS ENUM ('opened', 'wait', 'completed', 'closed');
  CREATE TYPE author_type AS ENUM ('employee', 'contact', 'user');
  CREATE TYPE import_status AS ENUM ('in_progress', 'completed', 'failed');
  CREATE TYPE tag_type AS ENUM ('auto', 'okdesk', 'manual');
  ```

### 4. Миграции - Основные таблицы

- [ ] Создать миграцию `YYYYMMDD_core_tables.sql`

  **Таблица sources:**
  ```sql
  CREATE TABLE sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    type source_type NOT NULL,
    config JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  ```

  **Таблица issues:**
  ```sql
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
  ```

  **Таблица messages:**
  ```sql
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
  ```

  **Таблица intents (динамическая):**
  ```sql
  CREATE TABLE intents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    code TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  ```

  **Таблица message_analysis:**
  ```sql
  CREATE TABLE message_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID REFERENCES messages(id) UNIQUE,
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reasoning TEXT
  );
  ```

  **Таблица message_intents:**
  ```sql
  CREATE TABLE message_intents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_analysis_id UUID REFERENCES message_analysis(id),
    intent_id UUID REFERENCES intents(id),
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    UNIQUE(message_analysis_id, intent_id)
  );
  ```

  **Таблица tags:**
  ```sql
  CREATE TABLE tags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    type tag_type,
    source VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  ```

  **Таблица message_tags:**
  ```sql
  CREATE TABLE message_tags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_analysis_id UUID REFERENCES message_analysis(id),
    tag_id UUID REFERENCES tags(id),
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    UNIQUE(message_analysis_id, tag_id)
  );
  ```

  **Таблица intent_clusters:**
  ```sql
  CREATE TABLE intent_clusters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    description TEXT,
    pattern TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  ```

  **Таблица message_clusters:**
  ```sql
  CREATE TABLE message_clusters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID REFERENCES messages(id),
    cluster_id UUID REFERENCES intent_clusters(id),
    similarity_score FLOAT CHECK (similarity_score >= 0 AND similarity_score <= 1),
    UNIQUE(message_id, cluster_id)
  );
  ```

  **Таблица imports:**
  ```sql
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
  ```

### 5. Индексы

- [ ] Создать миграцию `YYYYMMDD_indexes.sql`
  ```sql
  CREATE INDEX idx_messages_issue_id ON messages(issue_id);
  CREATE INDEX idx_messages_published_at ON messages(published_at);
  CREATE INDEX idx_message_analysis_message_id ON message_analysis(message_id);
  CREATE INDEX idx_issues_external_id ON issues(external_id);
  CREATE INDEX idx_issues_created_at ON issues(created_at);
  CREATE INDEX idx_message_intents_intent_id ON message_intents(intent_id);
  CREATE INDEX idx_message_tags_tag_id ON message_tags(tag_id);
  CREATE INDEX idx_intents_code ON intents(code);
  ```

### 6. Расширения PostgreSQL

- [ ] Создать миграцию для pg_trgm (полнотекстовый поиск)
  ```sql
  CREATE EXTENSION IF NOT EXISTS pg_trgm;
  CREATE INDEX idx_messages_content_trgm ON messages USING gin (content gin_trgm_ops);
  ```

### 7. Тестирование

- [ ] Запустить БД в Docker
  ```bash
  cd db
  make build
  make run
  ```

- [ ] Применить миграции
  ```bash
  make migrate
  ```

- [ ] Проверить схему
  ```bash
  docker exec -it support-db psql -U support -d support -c "\dt"
  docker exec -it support-db psql -U support -d support -c "\dT"
  ```

- [ ] Протестировать подключение из другого контейнера

- [ ] Протестировать INSERT/SELECT для каждой таблицы

### 8. Документация

- [ ] Обновить `db/README.md`
  - Описание структуры
  - Команды для работы
  - Описание ролей

- [ ] Создать диаграмму БД (ERD)

---

## Критерии готовности

- [x] БД запускается в Docker
- [ ] Все миграции применяются без ошибок
- [ ] Все таблицы созданы с правильными типами
- [ ] Индексы созданы
- [ ] Роли и права настроены
- [ ] Можно подключиться из других сервисов
- [ ] Документация актуальна

---

## Зависимости

**Блокирует:**
- Parser Service
- Analyzer Service
- Query Service

**Требует:**
- Docker установлен
- docker-compose.yml настроен
