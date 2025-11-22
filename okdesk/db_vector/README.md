# ChromaDB Vector Database

Векторное хранилище для embeddings issues системы анализа намерений пользователей.

## Описание

ChromaDB используется для:
- Хранения векторных представлений (embeddings) preprocessed issues
- Семантического поиска похожих обращений
- Кластеризации issues по схожести содержания

## Быстрый старт

```bash
# Запустить ChromaDB
make run

# Посмотреть логи
make logs

# Проверить статус и здоровье
make status
make health

# Остановить
make stop

# Перезапустить
make restart

# Полная очистка (удалить данные)
make clean
```

## Endpoints

- **HTTP API:** http://localhost:8100
- **Heartbeat:** http://localhost:8100/api/v1/heartbeat
- **Collections:** http://localhost:8100/api/v1/collections

## Volumes

- `support-chromadb-data` - персистентное хранилище векторов

## Конфигурация

Настройки в `.env`:

```env
CHROMA_SERVER_HOST=0.0.0.0
CHROMA_SERVER_HTTP_PORT=8000  # внутренний порт контейнера
IS_PERSISTENT=TRUE             # сохранение данных на диск
ANONYMIZED_TELEMETRY=FALSE     # отключение телеметрии
ALLOW_RESET=TRUE               # разрешить сброс коллекций
```

**Внешний порт:** `8100` (маппинг 8100:8000)

## Docker Network

ChromaDB использует сеть `support-network` для взаимодействия с другими сервисами:
- `support-db` (PostgreSQL)
- `support-analyzer` (Analyzer Service)
- `support-parser` (Parser Service)

Сеть создается автоматически при первом запуске через `make run`.

## Интеграция с Analyzer Service

Analyzer Service подключается к ChromaDB для:

1. **Stage 3 Pipeline** - сохранение embeddings после генерации
2. **Semantic Search** - поиск похожих issues
3. **Clustering** - загрузка всех embeddings для кластеризации

Пример подключения из Python:

```python
import chromadb

client = chromadb.HttpClient(host="chromadb", port=8000)
collection = client.get_or_create_collection("support_issues")

# Добавление embeddings
collection.add(
    ids=["issue-uuid-1", "issue-uuid-2"],
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    documents=["preprocessed text 1", "preprocessed text 2"]
)

# Поиск похожих
results = collection.query(
    query_embeddings=[[0.1, 0.2, ...]],
    n_results=10
)
```

## Troubleshooting

### ChromaDB не отвечает

```bash
# Проверить статус контейнера
make status

# Посмотреть логи
make logs

# Проверить healthcheck
make health
```

### Ошибка "network not found"

```bash
# Создать сеть вручную
make network
```

### Очистка данных

```bash
# Удалить все данные и начать с нуля
make clean
make run
```

## Production Notes

В production окружении рекомендуется:

1. **Бэкапы** - регулярное резервное копирование volume `support-chromadb-data`
2. **Мониторинг** - настроить алерты на healthcheck
3. **Ограничения ресурсов** - добавить limits в docker-compose:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2.0'
         memory: 4G
   ```
4. **Authentication** - добавить защиту API через reverse proxy (nginx)

## Полезные команды

```bash
# Просмотр всех коллекций
curl http://localhost:8100/api/v1/collections

# Количество векторов в коллекции
curl http://localhost:8100/api/v1/collections/support_issues/count

# Версия ChromaDB
curl http://localhost:8100/api/v1/version
```
