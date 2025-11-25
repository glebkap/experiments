# План реализации GUI Service

**task_id:** `07-gui-service`
**Дата:** 25.11.2025
**Приоритет:** Низкий (последний в roadmap)
**Статус:** 📋 Планирование

---

## 🎯 Цель

Реализовать веб-интерфейс (SPA) на React 18+ с TypeScript для системы анализа намерений пользователей службы поддержки.

## 🔧 Технологический стек

- **Framework:** React 18+ с TypeScript
- **Build Tool:** Vite
- **UI Library:** Material-UI (MUI)
- **Charts:** Recharts
- **State Management:** Context API
- **Routing:** React Router v6
- **HTTP Client:** Axios
- **Date Handling:** date-fns
- **Forms:** React Hook Form (опционально)
- **Code Quality:** ESLint + Prettier

## 📦 Структура проекта

```
services/gui/
├── frontend/
│   ├── src/
│   │   ├── api/                 # API client и типы
│   │   │   ├── client.ts
│   │   │   ├── endpoints.ts
│   │   │   └── types.ts
│   │   ├── components/          # Переиспользуемые компоненты
│   │   │   ├── common/
│   │   │   │   ├── Header.tsx
│   │   │   │   ├── Navigation.tsx
│   │   │   │   ├── LoadingSpinner.tsx
│   │   │   │   └── ErrorMessage.tsx
│   │   │   ├── tables/
│   │   │   │   └── DataTable.tsx
│   │   │   ├── charts/
│   │   │   │   ├── PieChartWrapper.tsx
│   │   │   │   ├── BarChartWrapper.tsx
│   │   │   │   ├── LineChartWrapper.tsx
│   │   │   │   └── WordCloud.tsx
│   │   │   └── upload/
│   │   │       └── FileUpload.tsx
│   │   ├── pages/               # Страницы приложения
│   │   │   ├── Dashboard/
│   │   │   │   └── Dashboard.tsx
│   │   │   ├── Import/
│   │   │   │   └── Import.tsx
│   │   │   ├── Issues/
│   │   │   │   ├── Issues.tsx
│   │   │   │   └── IssueDetail.tsx
│   │   │   ├── Analytics/
│   │   │   │   └── Analytics.tsx
│   │   │   ├── Clusters/
│   │   │   │   ├── Clusters.tsx
│   │   │   │   └── ClusterDetail.tsx
│   │   │   ├── Search/
│   │   │   │   └── Search.tsx
│   │   │   └── Export/
│   │   │       └── Export.tsx
│   │   ├── contexts/            # React Context
│   │   │   ├── ThemeContext.tsx
│   │   │   └── ApiContext.tsx
│   │   ├── hooks/               # Custom hooks
│   │   │   ├── useApi.ts
│   │   │   ├── useDebounce.ts
│   │   │   └── usePagination.ts
│   │   ├── utils/               # Утилиты
│   │   │   ├── formatters.ts
│   │   │   ├── validators.ts
│   │   │   └── constants.ts
│   │   ├── types/               # TypeScript типы
│   │   │   └── index.ts
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   └── vite-env.d.ts
│   ├── public/
│   │   └── favicon.ico
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   ├── vite.config.ts
│   ├── .eslintrc.cjs
│   └── .prettierrc
├── Dockerfile                   # Multi-stage build
├── nginx.conf                   # Nginx для production
└── README.md
```

---

## 📝 Пошаговый план реализации

### ✅ Этап 0: Подготовка окружения

**Цель:** Инициализировать проект, настроить инструменты разработки

**Задачи:**

0.1. ✅ Создать структуру директорий
0.2. ✅ Инициализировать Vite проект с React + TypeScript
0.3. ✅ Установить основные зависимости
0.4. ✅ Настроить TypeScript конфигурацию
0.5. ✅ Настроить ESLint и Prettier
0.6. ✅ Создать .gitignore
0.7. ✅ Проверить, что dev сервер запускается

**Критерии готовности:**
- ✅ Проект успешно инициализирован
- ✅ `npm run dev` запускает dev server
- ✅ Отображается стартовая страница React

---

### ✅ Этап 1: API Client и типы

**Цель:** Создать типизированный API клиент для работы с backend

**Задачи:**

1.1. Создать TypeScript типы для всех моделей данных
1.2. Реализовать Axios клиент с baseURL и interceptors
1.3. Создать типизированные функции для всех endpoints:
   - Parser endpoints (import)
   - Analyzer endpoints (pipeline, clustering, search)
   - Query endpoints (issues, stats, export)
1.4. Добавить обработку ошибок и retry логику
1.5. Написать mock данные для разработки (пока нет API Gateway)

**Файлы:**
- `src/api/types.ts` - все TypeScript интерфейсы
- `src/api/client.ts` - Axios instance
- `src/api/endpoints.ts` - функции для вызова API
- `src/api/mocks.ts` - mock данные

**Критерии готовности:**
- ✅ Все API типы определены
- ✅ Axios клиент настроен
- ✅ Mock данные доступны для разработки

---

### ✅ Этап 2: Базовая инфраструктура

**Цель:** Настроить routing, contexts, базовые компоненты

**Задачи:**

2.1. Настроить React Router v6 с маршрутами для всех страниц
2.2. Создать ThemeContext для темной/светлой темы
2.3. Создать ApiContext для управления API состоянием
2.4. Реализовать базовый Layout (Header + Navigation + Content)
2.5. Создать компонент Header с навигацией
2.6. Создать компонент Navigation (sidebar)
2.7. Создать LoadingSpinner и ErrorMessage компоненты

**Файлы:**
- `src/App.tsx` - routing setup
- `src/contexts/ThemeContext.tsx`
- `src/contexts/ApiContext.tsx`
- `src/components/common/Header.tsx`
- `src/components/common/Navigation.tsx`
- `src/components/common/LoadingSpinner.tsx`
- `src/components/common/ErrorMessage.tsx`

**Критерии готовности:**
- ✅ Routing работает между страницами
- ✅ Layout отображается корректно
- ✅ Навигация функциональна

---

### ✅ Этап 3: Custom Hooks и утилиты

**Цель:** Создать переиспользуемые hooks и helper функции

**Задачи:**

3.1. Реализовать `useApi` hook для API запросов (loading, error, data states)
3.2. Реализовать `useDebounce` hook для поиска
3.3. Реализовать `usePagination` hook
3.4. Создать formatters (date, confidence, text truncate)
3.5. Создать validators (file types, sizes)
3.6. Создать константы (API URLs, pagination defaults, chart colors)

**Файлы:**
- `src/hooks/useApi.ts`
- `src/hooks/useDebounce.ts`
- `src/hooks/usePagination.ts`
- `src/utils/formatters.ts`
- `src/utils/validators.ts`
- `src/utils/constants.ts`

**Критерии готовности:**
- ✅ Все hooks работают корректно
- ✅ Formatters покрыты тестами

---

### ✅ Этап 4: Переиспользуемые компоненты

**Цель:** Создать библиотеку UI компонентов

**Задачи:**

4.1. Реализовать DataTable с:
   - Сортировкой по колонкам
   - Пагинацией
   - Фильтрацией
   - Выбором строк
4.2. Создать FileUpload с drag-n-drop
4.3. Создать обертки для Recharts:
   - PieChartWrapper
   - BarChartWrapper
   - LineChartWrapper
   - WordCloud (использовать react-wordcloud)
4.4. Создать ConfidenceBar для отображения confidence scores
4.5. Создать TagBadge для отображения тегов

**Файлы:**
- `src/components/tables/DataTable.tsx`
- `src/components/upload/FileUpload.tsx`
- `src/components/charts/*.tsx`
- `src/components/common/ConfidenceBar.tsx`
- `src/components/common/TagBadge.tsx`

**Критерии готовности:**
- ✅ Все компоненты работают изолированно
- ✅ Компоненты переиспользуемы

---

### ✅ Этап 5: Страница Dashboard

**Цель:** Главная страница с общей статистикой

**Задачи:**

5.1. Создать layout Dashboard
5.2. Добавить карточки со статистикой:
   - Total issues
   - Total messages
   - Analyzed messages
   - Processing rate
5.3. Добавить PieChart с распределением по намерениям (топ-10)
5.4. Добавить LineChart временной динамики (issues по дням)
5.5. Добавить таблицу последних импортов
5.6. Добавить список топовых тегов (tag cloud или badges)
5.7. Подключить к API (или мокам)

**Файлы:**
- `src/pages/Dashboard/Dashboard.tsx`
- `src/pages/Dashboard/StatsCard.tsx`
- `src/pages/Dashboard/RecentImports.tsx`

**Критерии готовности:**
- ✅ Dashboard отображает все секции
- ✅ Графики рендерятся корректно
- ✅ Данные загружаются из API/моков

---

### ✅ Этап 6: Страница Import

**Цель:** Загрузка файлов и история импортов

**Задачи:**

6.1. Создать layout Import
6.2. Добавить FileUpload компонент с drag-n-drop
6.3. Добавить выбор типа источника (OKDesk/Telegram) через Radio buttons
6.4. Реализовать логику загрузки файла:
   - Валидация (JSONL для OKDesk, JSON для Telegram)
   - Прогресс бар
   - Обработка ошибок
6.5. Добавить таблицу истории импортов с фильтрами:
   - По статусу (in_progress, completed, failed)
   - По типу источника
   - По дате
6.6. Добавить модальное окно с деталями импорта (stats, errors)
6.7. Подключить к Parser Service API

**Файлы:**
- `src/pages/Import/Import.tsx`
- `src/pages/Import/UploadForm.tsx`
- `src/pages/Import/ImportHistory.tsx`
- `src/pages/Import/ImportDetailModal.tsx`

**Критерии готовности:**
- ✅ Можно загрузить файл
- ✅ Отображается прогресс загрузки
- ✅ История импортов работает
- ✅ Модальное окно показывает детали

---

### ✅ Этап 7: Страница Issues

**Цель:** Список обращений с фильтрацией

**Задачи:**

7.1. Создать layout Issues
7.2. Реализовать DataTable с колонками:
   - ID (short UUID)
   - Title
   - Status (badge)
   - Priority (1-4)
   - Source (OKDesk/Telegram)
   - Created At
   - Actions (View button)
7.3. Добавить фильтры:
   - По статусу (opened, wait, completed, closed)
   - По источнику (multi-select)
   - По диапазону дат (date range picker)
7.4. Добавить полнотекстовый поиск (debounced)
7.5. Добавить сортировку по всем колонкам
7.6. Добавить пагинацию
7.7. Клик на строку → переход к IssueDetail
7.8. Подключить к Query Service API

**Файлы:**
- `src/pages/Issues/Issues.tsx`
- `src/pages/Issues/IssuesTable.tsx`
- `src/pages/Issues/IssuesFilters.tsx`

**Критерии готовности:**
- ✅ Таблица отображает issues
- ✅ Фильтры работают
- ✅ Поиск работает с debounce
- ✅ Пагинация работает

---

### ✅ Этап 8: Страница IssueDetail

**Цель:** Детальная информация об обращении

**Задачи:**

8.1. Создать layout IssueDetail
8.2. Добавить карточку с информацией о issue:
   - Title, Description
   - Status, Priority
   - Source, External ID
   - Created At, Updated At, Completed At
8.3. Реализовать timeline сообщений (вертикальная лента)
8.4. Для каждого сообщения отобразить:
   - Автор (имя, тип), дата
   - Текст сообщения (с сохранением форматирования)
   - Намерения с ConfidenceBar
   - Теги с TagBadge
   - Reasoning (раскрывающийся Accordion)
8.5. Добавить кнопку "Переанализировать" для каждого сообщения
8.6. Добавить кнопку "Назад к списку"
8.7. Подключить к Query Service API

**Файлы:**
- `src/pages/Issues/IssueDetail.tsx`
- `src/pages/Issues/IssueCard.tsx`
- `src/pages/Issues/MessageTimeline.tsx`
- `src/pages/Issues/MessageCard.tsx`

**Критерии готовности:**
- ✅ Карточка issue отображается
- ✅ Timeline сообщений работает
- ✅ Намерения и теги видны
- ✅ Reasoning раскрывается

---

### ✅ Этап 9: Страница Analytics

**Цель:** Аналитика и визуализация данных

**Задачи:**

9.1. Создать layout Analytics с вкладками (Tabs):
   - Намерения
   - Теги
   - Источники
   - Временная динамика
9.2. **Вкладка Намерения:**
   - PieChart с топ-10 намерений
   - BarChart со всеми намерениями (sorted by count)
   - Таблица с деталями (name, count, confidence avg)
9.3. **Вкладка Теги:**
   - WordCloud для топ-50 тегов
   - BarChart с топ-20 тегами
   - Таблица тегов с фильтрацией по типу (auto/okdesk/manual)
9.4. **Вкладка Источники:**
   - PieChart распределения issues по источникам
   - BarChart сравнения (issues count, messages count)
   - Таблица статистики по источникам
9.5. **Вкладка Временная динамика:**
   - LineChart issues по дням (с фильтром по периоду)
   - LineChart messages по дням
   - BarChart активности по часам/дням недели
9.6. Подключить к Query Service API (stats endpoints)

**Файлы:**
- `src/pages/Analytics/Analytics.tsx`
- `src/pages/Analytics/IntentsTab.tsx`
- `src/pages/Analytics/TagsTab.tsx`
- `src/pages/Analytics/SourcesTab.tsx`
- `src/pages/Analytics/TimelineTab.tsx`

**Критерии готовности:**
- ✅ Все вкладки работают
- ✅ Графики отображаются корректно
- ✅ Фильтры применяются

---

### ✅ Этап 10: Страница Clusters

**Цель:** Просмотр кластеров схожих обращений

**Задачи:**

10.1. Создать layout Clusters
10.2. Отобразить список кластеров в виде карточек (Grid):
   - Название кластера (если есть)
   - Описание (если есть)
   - Количество сообщений
   - Топ-5 тегов кластера
   - Average distance to centroid
10.3. Добавить сортировку кластеров (by size, by density)
10.4. Клик на карточку → переход к ClusterDetail
10.5. **ClusterDetail страница:**
   - Информация о кластере
   - Таблица сообщений в кластере
   - Фильтры (by distance, by tag)
   - Возможность перейти к IssueDetail
10.6. Подключить к Analyzer Service API (clustering endpoints)

**Файлы:**
- `src/pages/Clusters/Clusters.tsx`
- `src/pages/Clusters/ClusterCard.tsx`
- `src/pages/Clusters/ClusterDetail.tsx`

**Критерии готовности:**
- ✅ Список кластеров отображается
- ✅ Детали кластера работают
- ✅ Навигация работает

---

### ✅ Этап 11: Страница Search

**Цель:** Полнотекстовый поиск с фильтрами

**Задачи:**

11.1. Создать layout Search
11.2. Добавить строку поиска (TextField с debounce)
11.3. Добавить расширенные фильтры (раскрывающаяся панель):
   - Multi-select для намерений
   - Multi-select для тегов
   - Date range picker
   - Source type selector
   - Status selector
11.4. Отобразить результаты поиска в виде карточек:
   - Issue title + snippet
   - Подсветка найденных слов (highlight)
   - Показать источник, дату, статус
   - Кнопка "Открыть" → IssueDetail
11.5. Добавить пагинацию результатов
11.6. Добавить функцию "Сохраненные запросы" (localStorage):
   - Сохранить текущий запрос с фильтрами
   - Список сохраненных запросов
   - Быстрая загрузка сохраненного запроса
11.7. Подключить к Query Service API (fulltext search)

**Файлы:**
- `src/pages/Search/Search.tsx`
- `src/pages/Search/SearchBar.tsx`
- `src/pages/Search/SearchFilters.tsx`
- `src/pages/Search/SearchResults.tsx`
- `src/pages/Search/SavedQueries.tsx`

**Критерии готовности:**
- ✅ Поиск работает с debounce
- ✅ Фильтры применяются
- ✅ Результаты отображаются корректно
- ✅ Сохраненные запросы работают

---

### ✅ Этап 12: Страница Export

**Цель:** Экспорт данных в различных форматах

**Задачи:**

12.1. Создать layout Export
12.2. Добавить выбор формата экспорта (Radio buttons):
   - CSV
   - JSON
12.3. Добавить фильтры данных для экспорта (аналогично Search):
   - Намерения
   - Теги
   - Даты
   - Источники
   - Статусы
12.4. Добавить выбор полей для экспорта (Checkboxes):
   - Issue fields (id, title, description, status, etc.)
   - Message fields (content, author, date, etc.)
   - Analysis fields (intents, tags, reasoning)
12.5. Добавить предпросмотр (сколько записей будет экспортировано)
12.6. Реализовать кнопку "Скачать":
   - Показать прогресс генерации
   - Автоматическое скачивание файла
   - Обработка ошибок
12.7. Подключить к Query Service API (export endpoint)

**Файлы:**
- `src/pages/Export/Export.tsx`
- `src/pages/Export/ExportForm.tsx`
- `src/pages/Export/FieldSelector.tsx`
- `src/pages/Export/ExportPreview.tsx`

**Критерии готовности:**
- ✅ Выбор формата работает
- ✅ Фильтры применяются
- ✅ Файл скачивается
- ✅ Прогресс отображается

---

### ✅ Этап 13: Styling и UX

**Цель:** Улучшить визуальный дизайн и пользовательский опыт

**Задачи:**

13.1. Настроить Material-UI theme:
   - Кастомные цвета (primary, secondary)
   - Typography settings
   - Spacing и breakpoints
13.2. Реализовать темную/светлую тему через ThemeContext
13.3. Добавить переключатель темы в Header
13.4. Проверить responsive дизайн на всех страницах:
   - Mobile (< 600px)
   - Tablet (600-960px)
   - Desktop (> 960px)
13.5. Добавить transitions и animations (где уместно)
13.6. Добавить loading skeletons для таблиц и карточек
13.7. Улучшить accessibility (ARIA labels, keyboard navigation)

**Файлы:**
- `src/theme.ts` - Material-UI theme config
- `src/contexts/ThemeContext.tsx` - обновить

**Критерии готовности:**
- ✅ Темная/светлая тема работает
- ✅ Responsive дизайн везде
- ✅ Accessibility на хорошем уровне

---

### ✅ Этап 14: Тестирование

**Цель:** Покрыть тестами критичный функционал

**Задачи:**

14.1. Настроить Jest + React Testing Library
14.2. Написать unit тесты для:
   - Custom hooks (useApi, useDebounce, usePagination)
   - Formatters и validators
   - API client functions
14.3. Написать component тесты для:
   - DataTable
   - FileUpload
   - Charts wrappers
14.4. Написать integration тесты для:
   - Dashboard page
   - Issues page
   - Search page
14.5. (Опционально) Настроить E2E тесты с Playwright:
   - User flow: Import → View Issues → Search
   - User flow: Dashboard → Analytics

**Файлы:**
- `src/hooks/__tests__/*.test.ts`
- `src/utils/__tests__/*.test.ts`
- `src/components/__tests__/*.test.tsx`
- `src/pages/__tests__/*.test.tsx`

**Критерии готовности:**
- ✅ Покрытие unit тестами > 70%
- ✅ Все критичные компоненты покрыты

---

### ✅ Этап 15: Docker и Production Build

**Цель:** Подготовить production deployment

**Задачи:**

15.1. Настроить production build в Vite:
   - Оптимизация bundle size
   - Code splitting
   - Minification
15.2. Создать Dockerfile (multi-stage build):
   - Stage 1: Build (Node.js)
   - Stage 2: Serve (Nginx)
15.3. Создать nginx.conf:
   - Раздача статики
   - Fallback на index.html для SPA routing
   - Proxy для API (когда будет API Gateway)
15.4. Настроить environment variables:
   - VITE_API_BASE_URL
   - VITE_APP_NAME
15.5. Протестировать Docker build локально
15.6. Обновить docker-compose.yml (добавить gui service)

**Файлы:**
- `services/gui/Dockerfile`
- `services/gui/nginx.conf`
- `services/gui/.env.example`
- `docker-compose.yml` (обновить)

**Критерии готовности:**
- ✅ Production build собирается
- ✅ Docker image работает
- ✅ Nginx корректно раздает статику

---

### ✅ Этап 16: Документация

**Цель:** Создать документацию для разработчиков

**Задачи:**

16.1. Написать README.md:
   - Описание проекта
   - Установка и запуск (dev режим)
   - Структура проекта
   - Доступные скрипты
   - Environment variables
16.2. Документировать API client:
   - Все endpoints
   - Примеры использования
16.3. Создать Storybook (опционально):
   - Stories для всех переиспользуемых компонентов
16.4. Обновить основной README проекта (добавить GUI Service)

**Файлы:**
- `services/gui/README.md`
- `services/gui/API.md`
- `README.md` (корень проекта - обновить)

**Критерии готовности:**
- ✅ README полный и понятный
- ✅ API документирован
- ✅ Примеры использования есть

---

### ✅ Этап 17: Интеграция с API Gateway

**Цель:** Подключить frontend к реальному API Gateway

**Задачи:**

17.1. Обновить API client:
   - Заменить моки на реальные endpoints
   - Обновить baseURL на API Gateway
17.2. Протестировать все интеграции:
   - Import flow
   - Issues list и detail
   - Search
   - Analytics
   - Export
17.3. Обработать edge cases:
   - Нет данных
   - Ошибки API
   - Timeout
17.4. Добавить retry логику где необходимо
17.5. Финальное тестирование полного flow

**Критерии готовности:**
- ✅ Все API интеграции работают
- ✅ Error handling на месте
- ✅ GUI полностью функционален

---

## 📊 Критерии успеха всего проекта

### Функциональные требования

- ✅ Все 8 страниц реализованы и работают
- ✅ API интеграция полностью функциональна
- ✅ Import flow работает (OKDesk + Telegram)
- ✅ Issues можно просматривать, фильтровать, искать
- ✅ Analytics отображает корректные графики
- ✅ Clusters визуализируются правильно
- ✅ Search работает с debounce и фильтрами
- ✅ Export генерирует и скачивает файлы

### Нефункциональные требования

- ✅ Responsive дизайн (mobile, tablet, desktop)
- ✅ Темная/светлая тема работает
- ✅ Время загрузки страниц < 2 сек
- ✅ Production build < 2 MB (gzipped)
- ✅ Accessibility score > 90 (Lighthouse)
- ✅ Unit test coverage > 70%

### Технические требования

- ✅ TypeScript используется везде (no any types)
- ✅ ESLint и Prettier настроены
- ✅ Docker image собирается успешно
- ✅ Документация полная

---

## 🔗 Зависимости

**Блокирующие:**
- ⚠️ API Gateway (в разработке другими разработчиками)
- ✅ Analyzer Service (реализован)
- ✅ Parser Service (реализован)

**Рекомендуемые:**
- Query Service для оптимизации запросов (можно работать напрямую с Analyzer/Parser)

---

## ⏱️ Оценка времени

**По этапам:**
- Этапы 0-3 (Setup + Infrastructure): 1 день
- Этапы 4-5 (Components + Dashboard): 1 день
- Этапы 6-8 (Import + Issues + IssueDetail): 2 дня
- Этапы 9-10 (Analytics + Clusters): 2 дня
- Этапы 11-12 (Search + Export): 2 дня
- Этапы 13-14 (Styling + Testing): 1-2 дня
- Этапы 15-16 (Docker + Docs): 1 день
- Этап 17 (API Integration): 1 день

**Итого:** 11-12 дней разработки (при полной занятости)

---

## 📌 Примечания

1. **Mock данные:** Пока API Gateway не готов, используем mock данные для разработки UI
2. **Progressive enhancement:** Начинаем с базового функционала, затем улучшаем UX
3. **Code review:** Желательно после каждого этапа проводить code review
4. **Git workflow:** Использовать feature branches, коммиты в формате `[07-gui-service] описание`
5. **Performance:** Использовать React.memo, useMemo, useCallback где необходимо
6. **SEO:** Не требуется (внутреннее приложение)

---

## 🚀 Следующие шаги

1. ✅ Утвердить план с пользователем
2. ⏳ Начать с Этапа 0: Подготовка окружения
3. ⏳ Последовательно выполнять этапы с approval после каждого
4. ⏳ Создавать коммиты после значимых изменений
5. ⏳ Генерировать changelog после завершения всех этапов

---

**Готов к началу реализации! 🎯**
