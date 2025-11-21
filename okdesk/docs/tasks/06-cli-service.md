# CLI Service - Декомпозиция задач

**Сервис:** support-cli
**Приоритет:** Средний
**Технологии:** Python 3.12, Click/Typer, Rich, httpx, uv

---

## Обзор

CLI Service - интерфейс командной строки для работы с системой. Взаимодействует с API Gateway.

---

## Структура

```
services/cli/
├── src/
│   ├── commands/
│   │   ├── import_cmd.py
│   │   ├── analyze_cmd.py
│   │   ├── search_cmd.py
│   │   ├── stats_cmd.py
│   │   └── export_cmd.py
│   ├── client/
│   │   └── api_client.py
│   ├── output/
│   │   ├── formatters.py    # Rich tables, progress bars
│   │   └── styles.py
│   ├── config.py
│   └── main.py
├── tests/
├── setup.py                  # Для установки как пакет
├── pyproject.toml
└── README.md
```

---

## Задачи

### 1. Настройка проекта

- [ ] Создать структуру
- [ ] Инициализировать uv проект
- [ ] Добавить зависимости: click/typer, rich, httpx
- [ ] Создать setup.py для установки
  - Entry point: support-cli

### 2. Configuration

- [ ] config.py
  - API Gateway URL (из env или ~/.support-cli/config.yaml)
  - Default batch size
- [ ] Поддержка config файла

### 3. API Client

- [ ] api_client.py
  - Класс APIClient
  - Методы для всех API endpoints:
    - import_okdesk(file_path)
    - import_telegram(file_path)
    - get_import_status(import_id)
    - list_imports()
    - analyze_batch(message_ids, batch_size)
    - analyze_all()
    - search(query, filters)
    - get_issue(issue_id)
    - get_stats_intents()
    - get_stats_tags()
    - get_stats_timeline()
    - export_data(format, filters)
  - Обработка ошибок
  - Timeout handling

### 4. Commands

#### 4.1 Import Commands

- [ ] import_cmd.py
  - support-cli import okdesk <file>
  - support-cli import telegram <file>
  - support-cli import status <import_id>
  - support-cli import history
  - Прогресс-бар для импорта (если async status)
  - Вывод статистики

#### 4.2 Analyze Commands

- [ ] analyze_cmd.py
  - support-cli analyze all [--batch-size N]
  - support-cli analyze recent [N]
  - support-cli analyze issue <id>
  - support-cli reanalyze <id>
  - Прогресс-бар для batch анализа

#### 4.3 Search Commands

- [ ] search_cmd.py
  - support-cli list issues [--limit N] [--status STATUS]
  - support-cli show issue <id>
  - support-cli search <query>
  - support-cli search intent <intent>
  - support-cli search tag <tag>
  - Таблицы для результатов (Rich)

#### 4.4 Stats Commands

- [ ] stats_cmd.py
  - support-cli stats intents
  - support-cli stats tags
  - support-cli stats sources
  - support-cli stats timeline [--from DATE] [--to DATE] [--granularity day|week|month]
  - support-cli report clusters
  - Красивые таблицы и графики (Rich)

#### 4.5 Export Commands

- [ ] export_cmd.py
  - support-cli export csv --query QUERY --output FILE
  - support-cli export json --query QUERY --output FILE
  - support-cli export report <type> --output FILE
  - Прогресс экспорта

### 5. Output Formatting

- [ ] formatters.py
  - Функции для форматирования:
    - format_table(data, columns)
    - format_progress_bar(current, total)
    - format_json_pretty(data)
    - format_stats_table(stats)
  - Использовать Rich Tables

- [ ] styles.py
  - Цветовая схема
  - Стили для разных типов вывода

### 6. Main Entry Point

- [ ] main.py
  - Click/Typer app
  - Регистрация всех команд
  - Global options (--api-url, --verbose)
  - Error handling

### 7. Тестирование

- [ ] Unit тесты для команд (с моками API client)
- [ ] Интеграционные тесты с реальным API

### 8. Документация

- [ ] README.md
  - Установка
  - Конфигурация
  - Список всех команд с примерами

---

## Критерии готовности

- [ ] Все команды реализованы
- [ ] Красивый вывод через Rich
- [ ] API client работает
- [ ] Конфигурация через файл/env
- [ ] Можно установить через pip/uv
- [ ] Документация актуальна

---

## Зависимости

**Требует:**
- API Gateway готов
