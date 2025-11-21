# Обзор задач разработки

**Проект:** Система анализа намерений пользователей службы поддержки
**Дата:** 21.11.2025

---

## Документы с декомпозицией

1. **[01-database-service.md](01-database-service.md)** - Database Service (PostgreSQL)
2. **[02-parser-service.md](02-parser-service.md)** - Parser Service (импорт данных)
3. **[03-analyzer-service.md](03-analyzer-service.md)** - Analyzer Service (pydantic-ai)
4. **[04-query-service.md](04-query-service.md)** - Query Service (поиск, статистика, экспорт)
5. **[05-api-gateway-service.md](05-api-gateway-service.md)** - API Gateway
6. **[06-cli-service.md](06-cli-service.md)** - CLI Service
7. **[07-gui-service.md](07-gui-service.md)** - GUI Service (React)

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

**2. Parser Service**
- **Приоритет:** Высокий
- **Время:** 3-5 дней
- **Зависит от:** Database Service
- **Задачи:**
  - DDD структура (domain, application, infrastructure, interfaces)
  - OKDesk и Telegram парсеры
  - API для импорта
  - Дедупликация

**3. Analyzer Service**
- **Приоритет:** Критический
- **Время:** 5-7 дней
- **Зависит от:** Database Service
- **Задачи:**
  - DDD структура
  - pydantic-ai агент для анализа
  - Пакетная обработка (10-50 сообщений)
  - Динамическое создание намерений
  - Кластеризация (опционально)

**Результат:** Можно импортировать данные и анализировать их

---

### Фаза 3: Интерфейсы доступа (Параллельная разработка)

**4. Query Service**
- **Приоритет:** Средний
- **Время:** 3-4 дня
- **Зависит от:** Database, Analyzer
- **Задачи:**
  - DDD структура (read models)
  - Поиск и фильтрация
  - Статистика
  - Экспорт (CSV, JSON)

**5. API Gateway**
- **Приоритет:** Средний
- **Время:** 1-2 дня
- **Зависит от:** Parser, Analyzer, Query
- **Задачи:**
  - Маршрутизация к сервисам
  - Health checks
  - CORS, логирование

**Результат:** Единый API для доступа к системе

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
2. ✅ Parser Service (только OKDesk)
3. ✅ Analyzer Service (базовый анализ без кластеризации)
4. ✅ CLI Service (базовые команды: import, analyze, stats)

### MVP позволяет:
- Импортировать данные из OKDesk
- Автоматически анализировать сообщения
- Просматривать результаты через CLI
- Получать базовую статистику

**Время разработки MVP:** ~2-3 недели

---

## Полная версия (v1.0)

### Дополнительно к MVP:
5. ✅ Query Service (полный)
6. ✅ API Gateway
7. ✅ Parser Service (с поддержкой Telegram)
8. ✅ CLI Service (все команды)
9. ✅ GUI Service
10. ✅ Analyzer Service (с кластеризацией)

**Время разработки полной версии:** ~1.5-2 месяца

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
