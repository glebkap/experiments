# GUI Service - Декомпозиция задач

**Сервис:** support-gui
**Приоритет:** Низкий (последний)
**Технологии:** React 19, TypeScript 5.9, Material-UI 7, Vite 7
**Статус:** ✅ РЕАЛИЗОВАН (базовая функциональность)

---

## Обзор

GUI Service - веб-интерфейс для работы с системой. SPA приложение на React с Material-UI.

---

## Структура (реализована)

```
services/gui/
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   │   ├── client.ts           # Axios instance с interceptors
│   │   │   ├── endpoints.ts        # Все API endpoints
│   │   │   ├── types.ts            # TypeScript интерфейсы (291 строка)
│   │   │   └── mocks.ts            # Mock данные для разработки
│   │   ├── components/
│   │   │   ├── common/
│   │   │   │   ├── Header.tsx
│   │   │   │   ├── Navigation.tsx
│   │   │   │   ├── LoadingSpinner.tsx
│   │   │   │   ├── ErrorMessage.tsx
│   │   │   │   ├── ErrorBoundary.tsx
│   │   │   │   ├── StatusBadge.tsx
│   │   │   │   ├── ConfidenceBar.tsx
│   │   │   │   └── TagBadge.tsx
│   │   │   ├── charts/
│   │   │   │   ├── LineChartWrapper.tsx
│   │   │   │   ├── PieChartWrapper.tsx
│   │   │   │   └── BarChartWrapper.tsx
│   │   │   ├── tables/
│   │   │   │   └── DataTable.tsx
│   │   │   └── upload/
│   │   │       └── FileUpload.tsx
│   │   ├── contexts/
│   │   │   └── ThemeContext.tsx
│   │   ├── hooks/
│   │   │   ├── useApi.ts
│   │   │   ├── useDebounce.ts
│   │   │   └── usePagination.ts
│   │   ├── pages/
│   │   │   ├── Dashboard/Dashboard.tsx
│   │   │   ├── Import/Import.tsx
│   │   │   ├── Issues/Issues.tsx
│   │   │   ├── Issues/IssueDetail.tsx
│   │   │   ├── Analytics/Analytics.tsx
│   │   │   ├── Clusters/Clusters.tsx
│   │   │   ├── Search/Search.tsx
│   │   │   └── Export/Export.tsx
│   │   ├── utils/
│   │   │   ├── formatters.ts
│   │   │   └── constants.ts
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   └── theme.ts
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── .eslintrc.cjs
│   └── .prettierrc
├── Dockerfile              # Multi-stage build (Node + Nginx)
├── nginx.conf              # SPA routing + security headers
└── README.md
```

---

## Задачи

### 1. Настройка проекта ✅

- [x] Инициализировать React проект (Vite + TypeScript)
- [x] Установить зависимости:
  - react 19, react-dom, react-router-dom 7
  - Material-UI 7 (@mui/material, @mui/icons-material, @mui/lab)
  - axios
  - recharts (для графиков)
  - date-fns (для работы с датами)
  - react-hook-form
- [x] Настроить TypeScript (strict mode)
- [x] Настроить ESLint и Prettier
- [x] Создать Dockerfile (multi-stage build)
- [x] Создать nginx.conf для SPA

### 2. API Client ✅

- [x] api/client.ts
  - Axios instance с базовым URL
  - Interceptors для обработки ошибок
  - Функция createFormData для загрузки файлов

- [x] api/endpoints.ts
  - importAPI (uploadOKDesk, uploadTelegram, getImportStatus, getImports)
  - pipelineAPI (processPipeline, getPipelineStatus)
  - clusteringAPI (runClustering, getClusteringInfo)
  - searchAPI (searchSimilar, searchFulltext)
  - issuesAPI (getIssues, getIssueDetail, getClusterIssues)
  - statsAPI (getProcessingStats, getClusterStats, getSourceStats, getTimelineStats)
  - exportAPI (exportData)
  - healthAPI (checkHealth, checkParserHealth, checkAnalyzerHealth)

- [x] api/types.ts (291 строка)
  - Все TypeScript типы для API моделей
  - Enums: SourceType, IssueStatus, AuthorType, ImportStatus, TagType

- [x] api/mocks.ts
  - Mock данные для разработки без бэкенда

### 3. Components ✅

- [x] Header.tsx - шапка с переключением темы
- [x] Navigation.tsx - боковое меню (drawer)
- [x] DataTable.tsx - универсальная таблица с сортировкой/пагинацией/выбором
- [x] Charts (LineChartWrapper, PieChartWrapper, BarChartWrapper)
- [x] LoadingSpinner.tsx
- [x] ErrorMessage.tsx
- [x] ErrorBoundary.tsx - обработка критических ошибок React
- [x] FileUpload.tsx - drag-n-drop компонент
- [x] StatusBadge.tsx - статусы с цветами
- [x] ConfidenceBar.tsx - прогресс-бар уверенности
- [x] TagBadge.tsx - теги с типами

### 4. Pages ✅

#### 4.1 Dashboard ✅
- [x] Карточки со статистикой (total issues, processed, messages, rate)
- [x] Прогресс обработки
- [x] Таблица последних импортов

#### 4.2 Import ✅
- [x] Drag-n-drop зона для загрузки файлов
- [x] Radio buttons для выбора типа (OKDesk/Telegram)
- [x] Прогресс-бар импорта
- [x] Таблица истории импортов
- [x] Модальное окно с деталями импорта

#### 4.3 Issues ✅
- [x] Таблица с обращениями
- [x] Фильтры: статус, поиск
- [x] Полнотекстовый поиск (debounced)
- [x] Пагинация
- [x] Клик на строку → переход к деталям

#### 4.4 Issue Detail ✅
- [x] Карточка с информацией об обращении
- [x] Timeline сообщений (MUI Lab Timeline)
- [x] Для каждого сообщения:
  - Текст
  - Автор и дата
  - Намерения с confidence bars
  - Теги с badges
  - Reasoning

#### 4.5 Analytics ✅
- [x] Tabs для разных типов аналитики
- [x] По источникам (pie chart + таблица)
- [x] Временная динамика (line chart + таблица)

#### 4.6 Clusters ✅
- [x] Список кластеров (cards grid)
- [x] Для каждого: название, описание, count
- [x] Модальное окно с деталями кластера

#### 4.7 Search ✅
- [x] Строка полнотекстового поиска
- [x] Расширяемые фильтры (статус, даты)
- [x] Результаты поиска

#### 4.8 Export ✅
- [x] Выбор формата (CSV/JSON)
- [x] Фильтры (статус, даты)
- [x] Предпросмотр полей
- [x] Кнопка "Скачать"

### 5. Hooks ✅

- [x] useApi.ts - загрузка данных с loading/error/refetch/reset
- [x] useDebounce.ts - debounce для поиска
- [x] usePagination.ts - пагинация

### 6. Utils ✅

- [x] formatters.ts
  - formatDate, formatRelativeTime
  - formatConfidence
  - truncateText, formatFileSize, formatDuration
  - formatNumber, shortenUUID
- [x] constants.ts
  - CHART_COLORS, STATUS_COLORS, PRIORITY_COLORS
  - IMPORT_STATUS_COLORS, SOURCE_TYPES
  - DEBOUNCE_DELAYS, PAGE_SIZES

### 7. Routing ✅

- [x] React Router 7 настроен
  - / → Dashboard
  - /import → Import
  - /issues → Issues
  - /issues/:id → IssueDetail
  - /analytics → Analytics
  - /clusters → Clusters
  - /search → Search
  - /export → Export

### 8. State Management ✅

- [x] Context API (ThemeContext для темы)
- [x] Dark/Light mode с сохранением в localStorage

### 9. Styling ✅

- [x] Material-UI 7 theme (light/dark)
- [x] Responsive дизайн (Grid2)
- [x] Цветовые схемы для статусов

### 10. Тестирование ⏳

- [ ] Unit тесты для компонентов (React Testing Library)
- [ ] E2E тесты (Cypress или Playwright)

### 11. Build and Deploy ✅

- [x] Production build (Vite)
- [x] Dockerfile с Nginx для статики
- [x] Environment variables (VITE_API_BASE_URL)
- [x] Health check endpoint
- [x] Security headers в nginx

### 12. Документация ✅

- [x] README.md с инструкциями
- [x] .env.example

---

## Критерии готовности

- [x] Все страницы реализованы (8 из 8)
- [x] API клиент настроен (8 групп endpoints)
- [x] Responsive дизайн (MUI Grid2)
- [x] Графики отображаются корректно (Recharts)
- [x] Drag-n-drop работает
- [x] Production build собирается
- [x] Docker интеграция
- [ ] Подключение к реальному API (использует mock данные)
- [ ] Unit/E2E тесты

---

## Зависимости

**Требует:**
- ✅ API Gateway готов
- ✅ Analyzer Service (для данных)

**Интеграция:**
- Добавлен в docker-compose.yml
- Порт: 3000
- API URL: http://localhost:8000/api/v1

---

## Запуск

### Development
```bash
cd services/gui/frontend
npm install
npm run dev
```

### Production (Docker)
```bash
docker-compose up gui
```

### Build
```bash
cd services/gui/frontend
npm run build
```
