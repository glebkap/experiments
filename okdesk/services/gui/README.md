# GUI Service - Support Intent Analyzer

Веб-интерфейс для системы анализа намерений пользователей службы поддержки.

## 🚀 Технологии

- **React 18** с TypeScript
- **Material-UI (MUI)** - UI компоненты
- **Recharts** - графики и визуализация
- **React Router v6** - маршрутизация
- **Axios** - HTTP клиент
- **Vite** - сборка и dev server
- **Nginx** - production сервер

## 📁 Структура проекта

```
frontend/
├── src/
│   ├── api/              # API клиент и типы
│   │   ├── client.ts     # Axios instance
│   │   ├── endpoints.ts  # API функции
│   │   ├── types.ts      # TypeScript типы
│   │   └── mocks.ts      # Mock данные для разработки
│   ├── components/       # Переиспользуемые компоненты
│   │   ├── common/       # Общие компоненты (Header, Navigation, etc)
│   │   ├── tables/       # DataTable
│   │   ├── charts/       # Обертки для Recharts
│   │   └── upload/       # FileUpload
│   ├── contexts/         # React Context
│   │   └── ThemeContext.tsx
│   ├── hooks/            # Custom hooks
│   │   ├── useApi.ts
│   │   ├── useDebounce.ts
│   │   └── usePagination.ts
│   ├── pages/            # Страницы приложения
│   │   ├── Dashboard/
│   │   ├── Import/
│   │   ├── Issues/
│   │   ├── Analytics/
│   │   ├── Clusters/
│   │   ├── Search/
│   │   └── Export/
│   ├── utils/            # Утилиты
│   │   ├── formatters.ts
│   │   └── constants.ts
│   ├── theme.ts          # Material-UI тема
│   ├── App.tsx           # Главный компонент
│   └── main.tsx          # Entry point
├── public/
├── .env                  # Environment variables
├── package.json
├── tsconfig.json
└── vite.config.ts
```

## 🛠️ Разработка

### Требования

- Node.js 18+
- npm или yarn

### Установка

```bash
cd frontend
npm install
```

### Запуск dev сервера

```bash
npm run dev
```

Приложение будет доступно на `http://localhost:3000`

### Сборка для production

```bash
npm run build
```

Собранные файлы будут в директории `dist/`

### Проверка кода

```bash
# ESLint
npm run lint

# TypeScript проверка
npm run type-check
```

## 🐳 Docker

### Сборка образа

```bash
# Из директории services/gui
docker build -t support-gui .
```

### Запуск контейнера

```bash
docker run -d \
  --name support-gui \
  -p 3000:3000 \
  -e VITE_API_BASE_URL=http://localhost:8000/api/v1 \
  support-gui
```

### Docker Compose

```yaml
services:
  gui:
    build: ./services/gui
    container_name: support-gui
    ports:
      - "3000:3000"
    environment:
      - VITE_API_BASE_URL=http://api-gateway:8000/api/v1
    depends_on:
      - api-gateway
    networks:
      - support-network
```

## ⚙️ Конфигурация

### Environment Variables

Создайте файл `.env` на основе `.env.example`:

```bash
cp .env.example .env
```

Доступные переменные:

- `VITE_API_BASE_URL` - URL API Gateway (по умолчанию: `http://localhost:8000/api/v1`)
- `VITE_APP_NAME` - Название приложения (по умолчанию: "Support Intent Analyzer")

## 📄 Страницы

### 1. Dashboard (/)
- Общая статистика (issues, messages, processing rate)
- Графики прогресса обработки
- История последних импортов

### 2. Import (/import)
- Загрузка файлов (OKDesk JSONL, Telegram JSON)
- История импортов с детальной статистикой

### 3. Issues (/issues)
- Список всех обращений
- Фильтры по статусу, поиск
- Переход к деталям обращения

### 4. Issue Detail (/issues/:id)
- Полная информация об обращении
- Timeline сообщений
- Анализ намерений и тегов для каждого сообщения

### 5. Analytics (/analytics)
- Статистика по источникам (PieChart, таблица)
- Временная динамика (LineChart)

### 6. Clusters (/clusters)
- Сетка кластеров схожих обращений
- Детали кластера с примерами issues

### 7. Search (/search)
- Полнотекстовый поиск
- Фильтры (статус, даты)
- Результаты в виде карточек

### 8. Export (/export)
- Выбор формата (CSV/JSON)
- Фильтры для экспорта
- Превью и скачивание

## 🎨 Темная/Светлая тема

Переключение темы доступно через кнопку в Header. Выбранная тема сохраняется в localStorage.

## 🔌 API Integration

### Mock Data

По умолчанию используются mock данные из `src/api/mocks.ts`.

Для переключения на реальный API:
1. Установите `USE_MOCK_DATA = false` в `src/api/mocks.ts`
2. Убедитесь, что API Gateway доступен по адресу из `VITE_API_BASE_URL`

### API Endpoints

Все endpoints определены в `src/api/endpoints.ts`:

**Import:**
- `POST /import/okdesk` - загрузка OKDesk файла
- `POST /import/telegram` - загрузка Telegram файла
- `GET /imports` - история импортов

**Analyzer:**
- `GET /analyzer/issues` - список обращений
- `GET /analyzer/issues/:id` - детали обращения
- `POST /analyzer/search/similar` - семантический поиск
- `GET /analyzer/stats/*` - статистика
- `POST /analyzer/export` - экспорт данных

**Clustering:**
- `POST /analyzer/clustering/run` - запуск кластеризации
- `GET /analyzer/clustering/info` - список кластеров

## 🧪 Тестирование

```bash
# Unit тесты
npm run test

# Coverage
npm run test:coverage
```

## 📝 Code Style

Проект использует:
- **ESLint** для линтинга
- **Prettier** для форматирования
- **TypeScript strict mode**

Все компоненты строго типизированы, `any` использование минимизировано.

## 🌐 Браузеры

Поддерживаются современные версии:
- Chrome/Edge (последние 2 версии)
- Firefox (последние 2 версии)
- Safari (последние 2 версии)

## 🤝 Разработка

### Добавление новой страницы

1. Создайте компонент в `src/pages/YourPage/YourPage.tsx`
2. Добавьте route в `src/App.tsx`
3. Добавьте пункт меню в `src/components/common/Navigation.tsx`

### Добавление нового API endpoint

1. Определите типы в `src/api/types.ts`
2. Добавьте функцию в `src/api/endpoints.ts`
3. Используйте через `useApi` hook

### Создание переиспользуемого компонента

Размещайте в `src/components/` с соответствующей категорией:
- `common/` - общие UI элементы
- `tables/` - таблицы
- `charts/` - графики
- `upload/` - загрузка файлов

## 📞 Поддержка

Для вопросов и предложений создавайте issue в репозитории проекта.

## 📜 Лицензия

Проект является частью Support Intent Analyzer системы.
