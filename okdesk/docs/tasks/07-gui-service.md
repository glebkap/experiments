# GUI Service - Декомпозиция задач

**Сервис:** support-gui
**Приоритет:** Низкий (последний)
**Технологии:** React 18+, TypeScript, Material-UI/Ant Design

---

## Обзор

GUI Service - веб-интерфейс для работы с системой. SPA приложение на React.

---

## Структура

```
services/gui/
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   │   ├── client.ts
│   │   │   └── types.ts
│   │   ├── components/
│   │   │   ├── Header.tsx
│   │   │   ├── Navigation.tsx
│   │   │   ├── Table.tsx
│   │   │   ├── Charts.tsx
│   │   │   └── LoadingSpinner.tsx
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx
│   │   │   ├── Import.tsx
│   │   │   ├── Issues.tsx
│   │   │   ├── IssueDetail.tsx
│   │   │   ├── Analytics.tsx
│   │   │   ├── Clusters.tsx
│   │   │   ├── Search.tsx
│   │   │   └── Export.tsx
│   │   ├── hooks/
│   │   │   ├── useApi.ts
│   │   │   └── useDebounce.ts
│   │   ├── utils/
│   │   │   └── formatters.ts
│   │   ├── App.tsx
│   │   └── index.tsx
│   ├── public/
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
├── Dockerfile
└── README.md
```

---

## Задачи

### 1. Настройка проекта

- [ ] Инициализировать React проект (Vite + TypeScript)
- [ ] Установить зависимости:
  - react, react-dom, react-router-dom
  - Material-UI или Ant Design
  - axios
  - chart.js или recharts
  - date-fns (для работы с датами)
- [ ] Настроить TypeScript
- [ ] Настроить ESLint и Prettier
- [ ] Создать Dockerfile (multi-stage build)

### 2. API Client

- [ ] api/client.ts
  - Axios instance с базовым URL
  - Interceptors для обработки ошибок
  - Типизированные методы для всех endpoints

- [ ] api/types.ts
  - TypeScript типы для всех API моделей:
    - Issue, Message, Intent, Tag
    - ImportJob, ImportStats
    - AnalysisResult
    - SearchRequest, SearchResponse
    - StatsResponse

### 3. Components

- [ ] Header.tsx - шапка сайта
- [ ] Navigation.tsx - боковое меню
- [ ] Table.tsx - переиспользуемая таблица с сортировкой/фильтрами/пагинацией
- [ ] Charts.tsx - обертки для графиков (PieChart, BarChart, LineChart)
- [ ] LoadingSpinner.tsx
- [ ] ErrorMessage.tsx
- [ ] FileUpload.tsx - drag-n-drop компонент

### 4. Pages

#### 4.1 Dashboard

- [ ] Dashboard.tsx
  - Карточки со статистикой (total issues, messages, analyzed)
  - Круговая диаграмма намерений
  - Линейный график временной динамики
  - Облако тегов
  - Таблица последних импортов

#### 4.2 Import

- [ ] Import.tsx
  - Drag-n-drop зона для загрузки файлов
  - Radio buttons для выбора типа (OKDesk/Telegram)
  - Прогресс-бар импорта
  - Таблица истории импортов с фильтрами
  - Модальное окно с деталями импорта

#### 4.3 Issues

- [ ] Issues.tsx
  - Таблица с обращениями
  - Фильтры: статус, источник, период дат
  - Полнотекстовый поиск (debounced)
  - Сортировка по колонкам
  - Пагинация
  - Клик на строку → переход к деталям

#### 4.4 Issue Detail

- [ ] IssueDetail.tsx
  - Карточка с информацией об обращении
  - Timeline сообщений
  - Для каждого сообщения:
    - Текст
    - Автор и дата
    - Намерения с confidence bars
    - Теги с badges
    - Reasoning (раскрывающийся блок)
    - Кнопка "Переанализировать"

#### 4.5 Analytics

- [ ] Analytics.tsx
  - Tabs для разных типов аналитики:
    - Намерения (pie chart + bar chart)
    - Теги (word cloud + top 20 bar chart)
    - Источники (comparison bar chart)
    - Временная динамика (line chart с фильтрами)

#### 4.6 Clusters

- [ ] Clusters.tsx
  - Список кластеров (cards)
  - Для каждого: название, описание, count сообщений, top tags
  - Клик → детали кластера
  - ClusterDetail: список сообщений в кластере

#### 4.7 Search

- [ ] Search.tsx
  - Строка полнотекстового поиска
  - Расширенные фильтры (multi-select для intents/tags, date range)
  - Результаты с подсветкой найденных слов
  - Сохраненные запросы (localStorage)

#### 4.8 Export

- [ ] Export.tsx
  - Выбор формата (CSV/JSON)
  - Применение фильтров (как в Search)
  - Выбор полей для экспорта (checkboxes)
  - Кнопка "Скачать"
  - Прогресс экспорта

### 5. Hooks

- [ ] useApi.ts
  - Custom hook для API запросов
  - Loading, error, data states
  - Автоматический refetch

- [ ] useDebounce.ts
  - Debounce для поиска

### 6. Utils

- [ ] formatters.ts
  - formatDate(date)
  - formatConfidence(confidence) - в проценты
  - truncateText(text, maxLength)

### 7. Routing

- [ ] Настроить React Router
  - / → Dashboard
  - /import → Import
  - /issues → Issues
  - /issues/:id → IssueDetail
  - /analytics → Analytics
  - /clusters → Clusters
  - /search → Search
  - /export → Export

### 8. State Management

- [ ] Решить: Context API или Redux (рекомендация: Context API для простоты)
- [ ] Создать contexts для:
  - Auth (если будет в будущем)
  - Theme (dark/light mode)

### 9. Styling

- [ ] Настроить theme (Material-UI/Ant Design)
- [ ] Адаптивный дизайн (responsive)

### 10. Тестирование

- [ ] Unit тесты для компонентов (React Testing Library)
- [ ] E2E тесты (Cypress или Playwright)

### 11. Build and Deploy

- [ ] Настроить production build (Vite)
- [ ] Dockerfile с Nginx для статики
- [ ] Environment variables для API URL

### 12. Документация

- [ ] README.md
  - Установка и запуск
  - Структура проекта
  - Компоненты

---

## Критерии готовности

- [ ] Все страницы реализованы
- [ ] API интеграция работает
- [ ] Responsive дизайн
- [ ] Графики отображаются корректно
- [ ] Drag-n-drop работает
- [ ] Production build собирается

---

## Зависимости

**Требует:**
- API Gateway готов
- Query Service для данных
