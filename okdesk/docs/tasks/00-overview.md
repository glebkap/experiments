# Обзор задач разработки

**Проект:** Система анализа намерений пользователей службы поддержки
**Дата:** 25.11.2025 (обновлено)

---

## Документы с декомпозицией

1. **[01-database-service.md](01-database-service.md)** ✅ - Database Service (PostgreSQL)
2. **[02-parser-service.md](02-parser-service.md)** ✅ - Parser Service (импорт данных OKDesk/Telegram)
3. **[03-analyzer-service/plan.md](03-analyzer-service/plan.md)** ✅ - Analyzer Service (Pipeline + ML + ChromaDB)
4. **[04-analyzer-extensions.md](04-analyzer-extensions.md)** 🔄 - Расширение Analyzer (просмотр issues, статистика, экспорт)
5. ~~**04-query-service.md**~~ ❌ - Query Service (упразднен, функции перенесены в Analyzer)
6. **[05-api-gateway.md](05-api-gateway.md)** 🔄 - API Gateway (reverse proxy)
7. **[06-cli-service.md](06-cli-service.md)** - CLI Service
8. **[07-gui-service.md](07-gui-service.md)** - GUI Service (React)

---

## Порядок разработки

### Фаза 1: Инфраструктура и база данных (Критический путь)

**1. Database Service**
- **Приоритет:** Критический (блокирует все)
- **Время:** 1-2 дня
- **Задачи:**
  - Настройка PostgreSQL в Docker
  - Создание миграций
  - ENUM типы, таблицы, индексы
  - Тестирование подключения

**Результат:** База данных готова к использованию

---

### Фаза 2: Базовые сервисы (Параллельная разработка)

**2. Parser Service** ✅ ЗАВЕРШЕН
- **Приоритет:** Высокий
- **Статус:** Реализован
- **Зависит от:** Database Service
- **Реализовано:**
  - DDD структура (domain, application, infrastructure, interfaces)
  - OKDesk и Telegram парсеры
  - API для импорта (POST /api/v1/import/okdesk, POST /api/v1/import/telegram)
  - Дедупликация по external_id

**3. Analyzer Service** ✅ ЗАВЕРШЕН (базовая функциональность)
- **Приоритет:** Критический
- **Статус:** Реализован pipeline + кластеризация + семантический поиск
- **Зависит от:** Database Service, ChromaDB
- **Реализовано:**
  - DDD структура + Pipeline Processing (4 этапа)
  - Preprocessing (HTML cleanup, лемматизация)
  - Генерация эмбеддингов (SentenceTransformers)
  - ChromaDB для векторного поиска
  - Кластеризация (HDBSCAN/K-means)
  - Семантический поиск

**Результат:** Можно импортировать данные, обрабатывать через ML pipeline, кластеризовать и искать похожие issues

---

### Фаза 3: Расширение Analyzer и API Gateway

**4. Analyzer Service Extensions** 🔄 В РАЗРАБОТКЕ
- **Приоритет:** Высокий
- **Время:** 2-3 дня
- **Зависит от:** Analyzer Service (базовый)
- **Задачи:**
  - Просмотр issues с фильтрацией (GET /analyzer/issues)
  - Детали обращения (GET /analyzer/issues/{id})
  - Просмотр кластеров (GET /analyzer/clusters/{id}/issues)
  - Полнотекстовый поиск (GET /analyzer/search/fulltext)
  - Статистика (GET /analyzer/stats/*)
  - Экспорт данных (POST /analyzer/export)

**5. API Gateway** ✅ ЗАВЕРШЕН
- **Приоритет:** Средний
- **Статус:** Реализован
- **Зависит от:** Parser, Analyzer
- **Реализовано:**
  - Reverse proxy к Parser и Analyzer
  - Агрегация health checks всех сервисов
  - CORS middleware для веб-клиентов
  - Логирование всех proxy запросов
  - Unit тесты (18 тестов, 91% coverage)
  - Docker интеграция

**Результат:** Полнофункциональная система с единой точкой входа

---

### Фаза 4: Пользовательские интерфейсы

**6. CLI Service**
- **Приоритет:** Средний
- **Время:** 2-3 дня
- **Зависит от:** API Gateway
- **Задачи:**
  - Команды для всех операций
  - Rich форматирование вывода
  - Конфигурация через файл

**7. GUI Service**
- **Приоритет:** Низкий
- **Время:** 7-10 дней
- **Зависит от:** API Gateway
- **Задачи:**
  - React + TypeScript setup
  - Все страницы (Dashboard, Import, Issues, Analytics, etc)
  - Графики и визуализация
  - Responsive дизайн

**Результат:** Полноценный веб-интерфейс

---

## Минимально жизнеспособный продукт (MVP)

### MVP включает:

1. ✅ Database Service
2. ✅ Parser Service (OKDesk + Telegram)
3. ✅ Analyzer Service (pipeline + кластеризация + семантический поиск)
4. 🔄 Analyzer Extensions (просмотр issues, статистика)
5. 🔄 API Gateway (reverse proxy)
6. CLI Service (базовые команды)

### MVP позволяет:

- Импортировать данные из OKDesk и Telegram
- Обрабатывать issues через ML pipeline (preprocessing + embeddings)
- Кластеризовать схожие issues
- Семантический поиск похожих issues
- Просматривать результаты через API
- Получать статистику

**Статус MVP:** 85% завершен (осталось Analyzer Extensions + API Gateway)

---

## Полная версия (v1.0)

### Дополнительно к MVP:

7. CLI Service (полный функционал с Rich форматированием)
8. GUI Service (React web interface)
9. Расширенная аналитика и визуализация
10. Экспорт в различные форматы

---

## Рекомендуемый подход

### Week 1-2: Инфраструктура
- [ ] Database Service
- [ ] Docker Compose setup
- [ ] CI/CD базовый

### Week 3-4: Core Services
- [ ] Parser Service (OKDesk)
- [ ] Analyzer Service (базовый)
- [ ] Интеграционные тесты

### Week 5: Query & Gateway
- [ ] Query Service
- [ ] API Gateway
- [ ] Тестирование связки

### Week 6: CLI
- [ ] CLI Service
- [ ] Документация
- [ ] **MVP готов**

### Week 7-10: GUI
- [ ] Frontend setup
- [ ] Все страницы
- [ ] Интеграция с API
- [ ] E2E тесты

### Week 11-12: Доработки
- [ ] Parser Service (Telegram)
- [ ] Кластеризация
- [ ] Оптимизация производительности
- [ ] Финальное тестирование
- [ ] **v1.0 готова**

---

## Архитектурные принципы

Все backend сервисы используют **DDD (Domain-Driven Design)**:

### Слои DDD:

1. **Domain Layer**
   - Models (Entities, Value Objects)
   - Repositories (interfaces)
   - Domain Services

2. **Application Layer**
   - Use Cases
   - DTOs

3. **Infrastructure Layer**
   - Repository implementations
   - External services (LLM, parsers, HTTP clients)
   - Database (SQLAlchemy)

4. **Interface Layer**
   - API (FastAPI routes)
   - CLI (если есть)

### Dependency Injection:
- Все зависимости через конструкторы
- FastAPI Depends для DI
- Легко тестировать через моки

---

## Технологический стек

### Backend (все сервисы)
- **Python 3.12**
- **uv** - для управления зависимостями
- **FastAPI** - web framework
- **SQLAlchemy 2.0** - ORM
- **pytest** - тестирование

### Специфичные библиотеки
- **Parser:** BeautifulSoup4
- **Analyzer:** pydantic-ai, scikit-learn
- **CLI:** Click/Typer, Rich

### Frontend
- **React 18+** (TypeScript)
- **Material-UI** или **Ant Design**
- **Chart.js** или **Recharts**
- **Vite** - build tool

### Infrastructure
- **Docker** + **Docker Compose**
- **PostgreSQL 13+**

---

## Тестирование

### Уровни тестирования:

1. **Unit тесты**
   - Domain layer (бизнес-логика)
   - Application layer (use cases с моками)
   - Coverage > 80%

2. **Integration тесты**
   - Repository implementations с тестовой БД
   - API endpoints
   - Parser + DB

3. **E2E тесты**
   - CLI команды
   - GUI (Cypress/Playwright)

---

## Метрики успеха

### MVP:
- [ ] Импорт 1000+ обращений OKDesk
- [ ] Анализ завершается без ошибок
- [ ] Точность определения намерений > 70%
- [ ] CLI команды работают

### v1.0:
- [ ] Поддержка OKDesk и Telegram
- [ ] Точность определения намерений > 80%
- [ ] Веб-интерфейс работает
- [ ] Экспорт данных работает
- [ ] Кластеризация группирует схожие сообщения

---

## Документация

Создаваемые документы:

1. ✅ **PRD.md** - Product Requirements
2. ✅ **ARCHITECTURE.md** - Архитектура системы
3. ✅ **tasks/*.md** - Декомпозиция по сервисам
4. **README.md** - Общее описание и quick start
5. **API.md** - API документация (Swagger)
6. **DEPLOYMENT.md** - Инструкции по развертыванию
