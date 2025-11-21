# [db-001-setup] Настройка Database Service

**Дата**: 21.11.2025
**Приоритет**: Критический
**Статус**: Завершено ✅

## Описание

Выполнена полная настройка Database Service - базы данных PostgreSQL для системы анализа пользовательских намерений в обращениях службы поддержки.

## Что сделано

### 1. Структура и конфигурация

- Создан `Makefile` с командами для управления БД:
  - `make run` - запуск контейнера PostgreSQL
  - `make stop` - остановка контейнера
  - `make clean` - очистка данных и volumes
  - `make db-migrate` - применение миграций
  - `make db-rollback` - откат миграций
  - `make db-status` - статус миграций
  - `make new-db-migration` - создание новой миграции
  - `make install-tools` - установка dbmate

- Создан файл `.env` с параметрами подключения
- Создан `.gitignore` для исключения временных файлов

### 2. Миграции базы данных

Создано 5 миграций:

#### 20251121124800_enable_extensions.sql
- Установка расширения `pgcrypto` (для gen_random_uuid)

#### 20251121124809_enum_types.sql
- `source_type` ('okdesk', 'telegram')
- `issue_status` ('opened', 'wait', 'completed', 'closed')
- `author_type` ('employee', 'contact', 'user')
- `import_status` ('in_progress', 'completed', 'failed')
- `tag_type` ('auto', 'okdesk', 'manual')

#### 20251121124932_core_tables.sql
Создано 11 таблиц:
1. **sources** - конфигурация источников данных
2. **issues** - обращения в поддержку
3. **messages** - сообщения в обращениях
4. **intents** - каталог намерений (динамический)
5. **message_analysis** - результаты анализа сообщений
6. **message_intents** - связь сообщений и намерений (M:N)
7. **tags** - теги (авто, OKDesk, ручные)
8. **message_tags** - связь сообщений и тегов (M:N)
9. **intent_clusters** - кластеры схожих намерений
10. **message_clusters** - привязка сообщений к кластерам
11. **imports** - история импортов

Особенности:
- UUID primary keys для всех таблиц
- Foreign keys между связанными таблицами
- UNIQUE constraints для дедупликации
- CHECK constraints для валидации (confidence, priority)
- JSONB поля для гибкой структуры (config, stats)

#### 20251121125035_indexes.sql
Создано 8 индексов для оптимизации:
- `idx_messages_issue_id` - сообщения по обращению
- `idx_messages_published_at` - временная сортировка
- `idx_message_analysis_message_id` - поиск анализа
- `idx_issues_external_id` - дедупликация обращений
- `idx_issues_created_at` - сортировка обращений
- `idx_message_intents_intent_id` - группировка по намерениям
- `idx_message_tags_tag_id` - группировка по тегам
- `idx_intents_code` - поиск намерений по коду

#### 20251121125126_pg_extensions.sql
- Установка расширения `pg_trgm` для полнотекстового поиска
- GIN индекс `idx_messages_content_trgm` на поле `messages.content`

### 3. Тестирование

Проведено полное тестирование:
- ✅ Запуск контейнера PostgreSQL 12-alpine
- ✅ Применение всех 5 миграций
- ✅ Проверка создания таблиц (11 + schema_migrations)
- ✅ Проверка ENUM типов (5 типов)
- ✅ Проверка индексов (8 + автоматические)
- ✅ Проверка расширений (pgcrypto, pg_trgm)
- ✅ Тестирование INSERT/SELECT
- ✅ Тестирование Foreign Keys
- ✅ Тестирование UUID generation
- ✅ Тестирование полного workflow (source → issue → message → analysis → intent)

### 4. Документация

Создан подробный `db/README.md` с описанием:
- Быстрый старт
- Конфигурация БД
- Все команды Makefile
- Работа с миграциями
- Схема БД (типы, таблицы, индексы, расширения)
- Примеры подключения
- Архитектурные заметки
- Troubleshooting

## Технические детали

### Стек
- PostgreSQL 12-alpine
- dbmate v1.15.0 (для миграций)
- Docker (контейнеризация)

### Параметры БД
- Хост: localhost
- Порт: 5432
- База: postgres
- Пользователь: postgres
- Volume: support-db-pgdata

### Особенности реализации
- Все таблицы используют UUID как primary key
- Динамическое создание намерений LLM-агентом
- Множественные намерения на сообщение с confidence scores
- Дедупликация через composite unique constraints
- JSONB для гибкой конфигурации источников
- Полнотекстовый поиск через pg_trgm

## Файлы

```
db/
├── Makefile              # Команды управления БД
├── .env                  # Параметры подключения
├── .gitignore           # Исключения git
├── README.md            # Документация
└── migrations/          # Миграции
    ├── 20251121124800_enable_extensions.sql
    ├── 20251121124809_enum_types.sql
    ├── 20251121124932_core_tables.sql
    ├── 20251121125035_indexes.sql
    └── 20251121125126_pg_extensions.sql
```

## Следующие шаги

Database Service полностью готов для использования другими сервисами:
- ✅ Parser Service может начать импорт данных
- ✅ Analyzer Service может начать анализ сообщений
- ✅ Query Service может запрашивать данные

## Команды для работы

```bash
# Запуск БД
cd db
make run

# Применение миграций
make db-migrate

# Проверка статуса
make db-status

# Остановка
make stop

# Полная очистка
make clean
```

## Заметки

- База данных работает на стандартном порту 5432
- Все данные хранятся в Docker volume `support-db-pgdata`
- Миграции применяются автоматически через dbmate
- Для чистой установки: `make clean && make run && make db-migrate`
