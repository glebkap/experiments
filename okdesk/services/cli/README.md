# Support CLI

Интерфейс командной строки для системы анализа обращений службы поддержки.

## Возможности

- **Импорт данных** из OKDesk (JSONL) и Telegram (JSON)
- **Pipeline обработка** issues через ML pipeline
- **Кластеризация** схожих issues (HDBSCAN/K-means)
- **Семантический поиск** похожих issues
- **Полнотекстовый поиск** в PostgreSQL
- **Статистика** по обработке, кластерам, источникам
- **Экспорт данных** в CSV/JSON
- **Красивый вывод** с использованием Rich (таблицы, цвета, прогресс-бары)

## Установка

### Через uv (для разработки)

```bash
cd services/cli
uv pip install -e .
```

### Через Docker

```bash
docker build -t support-cli services/cli/
```

## Конфигурация

Создайте файл `.env` на основе `.env.example`:

```bash
cp .env.example .env
```

Основные параметры:

```env
API_GATEWAY_URL=http://localhost:8000
DEFAULT_BATCH_SIZE=100
REQUEST_TIMEOUT=300
LOG_LEVEL=INFO
```

## Использование

### Проверка здоровья API

```bash
support-cli health
```

### Импорт данных

**OKDesk:**

```bash
support-cli import okdesk /data/okdesk/out.jsonl
support-cli import okdesk /data/okdesk/out.jsonl --wait  # Ждать завершения
```

**Telegram:**

```bash
support-cli import telegram /data/telegram/result.json
```

**История импортов:**

```bash
support-cli import history --limit 10
support-cli import status <import-id>
```

### Pipeline обработка

**Запустить обработку:**

```bash
support-cli pipeline process --batch-size 100 --device cpu
```

**Проверить статус:**

```bash
support-cli pipeline status
```

### Кластеризация

**Запустить кластеризацию:**

```bash
support-cli cluster run --method hdbscan --min-size 5
support-cli cluster run --method hdbscan --use-llm  # С генерацией названий через LLM
```

**Просмотреть кластеры:**

```bash
support-cli cluster list
support-cli cluster show <cluster-id> --limit 20
```

### Поиск

**Семантический поиск:**

```bash
support-cli search similar "проблема с оплатой" --top-k 10
support-cli search similar "не работает" --min-similarity 0.8
```

**Полнотекстовый поиск:**

```bash
support-cli search fulltext "платеж не прошел" --limit 50
```

**Список issues:**

```bash
support-cli search list --status opened --limit 50
support-cli search list --source <source-id>
```

**Просмотр issue:**

```bash
support-cli search show <issue-id>
support-cli search show <issue-id> --format json
support-cli search show <issue-id> --no-messages  # Без сообщений
```

### Статистика

**Статистика обработки:**

```bash
support-cli stats processing
```

**Статистика кластеров:**

```bash
support-cli stats clusters
```

**Статистика по источникам:**

```bash
support-cli stats sources
```

**Временная динамика:**

```bash
support-cli stats timeline --from 2025-01-01 --to 2025-11-25 --granularity week
support-cli stats timeline --granularity month
```

### Экспорт

**Экспорт данных:**

```bash
support-cli export data --format csv --output export.csv --status opened
support-cli export data --format json --cluster <id> --output cluster.json
support-cli export data --format csv --from 2025-01-01 --to 2025-11-25
```

## Global Options

Все команды поддерживают глобальные опции:

```bash
support-cli --api-url http://custom-api:8000 [command]
support-cli --verbose [command]  # Подробный вывод
support-cli --quiet [command]    # Тихий режим
```

## Использование с Docker

### Запуск команды:

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -e API_GATEWAY_URL=http://host.docker.internal:8000 \
  support-cli import okdesk /app/data/okdesk/out.jsonl
```

### Использование docker-compose:

```bash
docker-compose run --rm cli import history
docker-compose run --rm cli pipeline status
docker-compose run --rm cli search similar "проблема"
```

## Примеры сценариев

### Полный цикл обработки данных

```bash
# 1. Импорт данных
support-cli import okdesk /data/okdesk/out.jsonl --wait

# 2. Запуск pipeline обработки
support-cli pipeline process --batch-size 100

# 3. Проверка статуса
support-cli pipeline status

# 4. Кластеризация
support-cli cluster run --method hdbscan --use-llm

# 5. Просмотр результатов
support-cli cluster list
support-cli stats processing
support-cli stats clusters
```

### Анализ конкретной проблемы

```bash
# 1. Поиск похожих issues
support-cli search similar "не могу оплатить" --top-k 20

# 2. Детали issue
support-cli search show <issue-id>

# 3. Поиск в том же кластере
support-cli cluster show <cluster-id> --limit 50

# 4. Экспорт для анализа
support-cli export data --format csv --cluster <cluster-id> --output analysis.csv
```

## Troubleshooting

### API недоступен

```bash
# Проверьте доступность API
support-cli health

# Проверьте URL в .env
cat .env | grep API_GATEWAY_URL

# Используйте другой URL
support-cli --api-url http://localhost:8000 health
```

### Ошибка импорта файла

```bash
# Проверьте формат файла
file /data/okdesk/out.jsonl

# Проверьте права доступа
ls -l /data/okdesk/out.jsonl

# Используйте абсолютный путь
support-cli import okdesk /absolute/path/to/file.jsonl
```

### Таймаут при обработке

```bash
# Увеличьте таймаут в .env
REQUEST_TIMEOUT=600

# Или через переменную окружения
REQUEST_TIMEOUT=600 support-cli pipeline process
```

## Development

### Запуск тестов:

```bash
cd services/cli
pytest
```

### Линтеры:

```bash
ruff check src/
mypy src/
```

### Форматирование:

```bash
ruff format src/
```

## Архитектура

```
services/cli/
├── src/support_cli/
│   ├── client/          # API client и модели
│   ├── commands/        # Typer команды
│   ├── output/          # Rich форматирование
│   ├── utils/           # Утилиты
│   ├── config.py        # Конфигурация
│   └── main.py          # Entry point
├── tests/               # Тесты
├── Dockerfile           # Docker образ
└── pyproject.toml       # Зависимости
```

## Зависимости

- **Python 3.12+**
- **Typer** - CLI framework
- **Rich** - красивый вывод
- **httpx** - HTTP client
- **Pydantic** - валидация данных

## Лицензия

Внутренний проект
