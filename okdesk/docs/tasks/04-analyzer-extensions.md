# Analyzer Service Extensions - Декомпозиция задач

**Task ID:** 04-analyzer-extensions
**Приоритет:** Высокий
**Дата:** 25.11.2025
**Зависит от:** Analyzer Service (базовый - завершен)

---

## Цель

Расширить Analyzer Service дополнительными endpoints для:
- Просмотра issues с фильтрацией
- Детального просмотра обращений
- Просмотра кластеров и их содержимого
- Полнотекстового поиска (PostgreSQL FTS)
- Статистики по различным параметрам
- Экспорта данных в CSV/JSON

**Важно:** Сервис НЕ должен содержать сложную бизнес-логику. Вся ML/математика уже реализована. Новые endpoints - это простые read-only операции с PostgreSQL.

---

## Архитектурные принципы

1. **Только чтение из БД** - новые endpoints не модифицируют данные
2. **Простые SQL запросы** - без сложной обработки, только SELECT с JOIN
3. **Использование существующих репозиториев** - расширяем имеющиеся или добавляем простые новые
4. **Pydantic schemas** - валидация входных/выходных данных
5. **Pagination** - для списков (limit/offset)

---

## Текущая структура Analyzer Service

```
services/analyzer/src/
├── domain/
│   ├── models/
│   ├── repositories/
│   └── services/
├── application/
│   ├── pipeline/
│   ├── use_cases/
│   └── dto/
├── infrastructure/
│   ├── persistence/postgres/
│   ├── ml/
│   └── vectordb/
└── interfaces/api/
    ├── routes.py          # Существующие endpoints
    ├── schemas.py
    └── __init__.py
```

---

## Задачи

### 1. Расширение Domain Layer

#### 1.1 Новые/обновленные Repository Methods

**`domain/repositories/issue_repository.py`** - добавить методы:

```python
@abstractmethod
async def get_issues_with_filters(
    self,
    status: Optional[str] = None,
    source_id: Optional[UUID] = None,
    priority: Optional[int] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    limit: int = 50,
    offset: int = 0
) -> List[Issue]:
    """Получить issues с фильтрацией и пагинацией"""
    pass

@abstractmethod
async def get_issue_with_messages(self, issue_id: UUID) -> Optional[Tuple[Issue, List[Message]]]:
    """Получить issue со всеми messages"""
    pass

@abstractmethod
async def count_issues_with_filters(
    self,
    status: Optional[str] = None,
    source_id: Optional[UUID] = None,
    priority: Optional[int] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None
) -> int:
    """Подсчет issues с фильтрами"""
    pass

@abstractmethod
async def fulltext_search(
    self,
    query: str,
    limit: int = 50,
    offset: int = 0
) -> List[Issue]:
    """Полнотекстовый поиск по preprocessed_issues (PostgreSQL FTS)"""
    pass
```

**`domain/repositories/cluster_repository.py`** - добавить методы:

```python
@abstractmethod
async def get_cluster_with_issues(
    self,
    cluster_id: UUID,
    limit: int = 50,
    offset: int = 0
) -> Optional[Tuple[Cluster, List[Issue]]]:
    """Получить кластер с issues"""
    pass

@abstractmethod
async def get_cluster_stats(self) -> List[Dict[str, Any]]:
    """Статистика по кластерам (size, avg_distance, etc)"""
    pass
```

**Создать `domain/repositories/stats_repository.py`:**

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any
from datetime import datetime

class StatsRepository(ABC):
    @abstractmethod
    async def get_processing_stats(self) -> Dict[str, Any]:
        """
        Статистика обработки:
        - total_issues
        - processed_issues
        - unprocessed_issues
        - total_embeddings
        """
        pass

    @abstractmethod
    async def get_sources_stats(self) -> List[Dict[str, Any]]:
        """
        Статистика по источникам:
        - source_id, name, type
        - issues_count
        - processed_count
        """
        pass

    @abstractmethod
    async def get_timeline_stats(
        self,
        date_from: datetime,
        date_to: datetime,
        group_by: str = "day"  # day, week, month
    ) -> List[Dict[str, Any]]:
        """
        Временная статистика:
        - date
        - issues_count
        - processed_count
        """
        pass
```

---

### 2. Infrastructure Layer - Repository Implementations

#### 2.1 Обновить `infrastructure/persistence/postgres/issue_repository_impl.py`

Реализовать новые методы:

```python
async def get_issues_with_filters(self, ...) -> List[Issue]:
    # SQL с WHERE filters + LIMIT/OFFSET
    # JOIN с sources для получения source_type
    pass

async def get_issue_with_messages(self, issue_id: UUID):
    # JOIN issues с messages
    # ORDER BY messages.published_at
    pass

async def fulltext_search(self, query: str, limit, offset):
    # Использовать GIN индекс: to_tsvector('russian', preprocessed_issues.content)
    # SELECT * FROM issues i
    # JOIN preprocessed_issues pi ON i.id = pi.id
    # WHERE to_tsvector('russian', pi.content) @@ plainto_tsquery('russian', :query)
    pass
```

#### 2.2 Обновить `infrastructure/persistence/postgres/cluster_repository_impl.py`

```python
async def get_cluster_with_issues(self, cluster_id, limit, offset):
    # JOIN clusters с issue_clusters с issues
    # ORDER BY distance_to_centroid ASC (ближайшие к центроиду)
    pass

async def get_cluster_stats(self):
    # GROUP BY cluster_id
    # Агрегаты: COUNT, AVG(distance_to_centroid), MIN, MAX
    pass
```

#### 2.3 Создать `infrastructure/persistence/postgres/stats_repository_impl.py`

```python
class StatsRepositoryImpl(StatsRepository):
    def __init__(self, db_session: AsyncSession, vectordb: VectorDBService):
        self.db = db_session
        self.vectordb = vectordb

    async def get_processing_stats(self) -> Dict[str, Any]:
        # COUNT queries на issues, preprocessed_issues
        # vectordb.count_total() для embeddings
        pass

    async def get_sources_stats(self):
        # GROUP BY source_id
        # JOIN sources, COUNT issues
        pass

    async def get_timeline_stats(self, date_from, date_to, group_by):
        # date_trunc(group_by, issues.created_at)
        # GROUP BY date
        pass
```

---

### 3. Application Layer - DTOs

**Создать `application/dto/issue_list_dto.py`:**

```python
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
from uuid import UUID

@dataclass
class IssueListItemDTO:
    id: str
    external_id: str
    title: Optional[str]
    status: str
    priority: Optional[int]
    created_at: datetime
    source_name: str
    source_type: str

@dataclass
class IssueListResponseDTO:
    items: List[IssueListItemDTO]
    total: int
    limit: int
    offset: int
```

**Создать `application/dto/issue_detail_dto.py`:**

```python
@dataclass
class MessageDTO:
    id: str
    external_id: str
    author_name: Optional[str]
    author_type: str
    content: str
    is_public: bool
    published_at: datetime

@dataclass
class IssueDetailDTO:
    id: str
    external_id: str
    title: Optional[str]
    description: Optional[str]
    status: str
    priority: Optional[int]
    created_at: datetime
    updated_at: Optional[datetime]
    completed_at: Optional[datetime]
    source_name: str
    source_type: str
    messages: List[MessageDTO]
    cluster_id: Optional[str]
    cluster_label: Optional[int]
```

**Создать `application/dto/stats_dto.py`:**

```python
@dataclass
class ProcessingStatsDTO:
    total_issues: int
    processed_issues: int
    unprocessed_issues: int
    total_embeddings: int
    total_clusters: int

@dataclass
class SourceStatsDTO:
    source_id: str
    name: str
    type: str
    issues_count: int
    processed_count: int

@dataclass
class TimelinePointDTO:
    date: str
    issues_count: int
    processed_count: int
```

---

### 4. Interface Layer - API Routes

**Обновить `interfaces/api/routes.py`** - добавить новые endpoints:

```python
# ==================== Issues Endpoints ====================
@router.get("/issues", response_model=IssueListResponseDTO)
async def get_issues(
    status: Optional[str] = None,
    source_id: Optional[str] = None,
    priority: Optional[int] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    issue_repo: IssueRepository = Depends(get_issue_repository)
):
    """
    Получить список issues с фильтрацией.

    Query Parameters:
    - status: opened, wait, completed, closed
    - source_id: UUID источника
    - priority: 1-4
    - date_from, date_to: ISO datetime
    - limit, offset: pagination
    """
    pass

@router.get("/issues/{issue_id}", response_model=IssueDetailDTO)
async def get_issue_detail(
    issue_id: str,
    issue_repo: IssueRepository = Depends(get_issue_repository),
    cluster_repo: ClusterRepository = Depends(get_cluster_repository)
):
    """
    Получить детальную информацию об issue с messages.
    """
    pass

@router.get("/clusters/{cluster_id}/issues", response_model=IssueListResponseDTO)
async def get_cluster_issues(
    cluster_id: str,
    limit: int = 50,
    offset: int = 0,
    cluster_repo: ClusterRepository = Depends(get_cluster_repository)
):
    """
    Получить issues в кластере, отсортированные по близости к центроиду.
    """
    pass

# ==================== Search Endpoints ====================
@router.get("/search/fulltext")
async def fulltext_search(
    q: str,
    limit: int = 50,
    offset: int = 0,
    issue_repo: IssueRepository = Depends(get_issue_repository)
):
    """
    Полнотекстовый поиск по preprocessed_issues (PostgreSQL FTS).

    Query Parameters:
    - q: поисковый запрос (текст)
    - limit, offset: pagination
    """
    pass

# ==================== Stats Endpoints ====================
@router.get("/stats/processing", response_model=ProcessingStatsDTO)
async def get_processing_stats(
    stats_repo: StatsRepository = Depends(get_stats_repository)
):
    """
    Общая статистика обработки.
    """
    pass

@router.get("/stats/sources", response_model=List[SourceStatsDTO])
async def get_sources_stats(
    stats_repo: StatsRepository = Depends(get_stats_repository)
):
    """
    Статистика по источникам данных.
    """
    pass

@router.get("/stats/clusters")
async def get_clusters_stats(
    cluster_repo: ClusterRepository = Depends(get_cluster_repository)
):
    """
    Статистика по кластерам.
    """
    pass

@router.get("/stats/timeline")
async def get_timeline_stats(
    date_from: str,
    date_to: str,
    group_by: str = "day",
    stats_repo: StatsRepository = Depends(get_stats_repository)
):
    """
    Временная статистика создания issues.

    Query Parameters:
    - date_from, date_to: ISO datetime
    - group_by: day, week, month
    """
    pass

# ==================== Export Endpoints ====================
@router.post("/export")
async def export_data(
    format: str = "csv",  # csv, json
    filters: Optional[Dict] = None,
    issue_repo: IssueRepository = Depends(get_issue_repository)
):
    """
    Экспорт данных в CSV или JSON.

    Body:
    - format: csv | json
    - filters: те же что и в GET /issues

    Returns:
    - StreamingResponse с файлом
    """
    pass
```

---

### 5. Обновление Pydantic Schemas

**Обновить `interfaces/api/schemas.py`** - добавить схемы валидации:

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class IssueFiltersRequest(BaseModel):
    status: Optional[str] = None
    source_id: Optional[str] = None
    priority: Optional[int] = Field(None, ge=1, le=4)
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: int = Field(50, ge=1, le=100)
    offset: int = Field(0, ge=0)

class ExportRequest(BaseModel):
    format: str = Field("csv", pattern="^(csv|json)$")
    filters: Optional[IssueFiltersRequest] = None
```

---

### 6. Dependency Injection

**Обновить `infrastructure/dependencies.py`** - добавить:

```python
def get_stats_repository(db: AsyncSession = Depends(get_db_session)) -> StatsRepository:
    return StatsRepositoryImpl(db, get_vector_db_service())
```

---

### 7. Тестирование

#### 7.1 Unit Tests

- Тесты для новых repository methods с моками
- Тесты для конвертации domain models в DTOs

#### 7.2 Integration Tests

- Тесты API endpoints с тестовой БД
- Проверка фильтрации, пагинации
- Проверка полнотекстового поиска

---

## Критерии готовности

- [ ] Все новые repository methods реализованы
- [ ] Все новые endpoints работают
- [ ] Фильтрация и пагинация корректны
- [ ] Полнотекстовый поиск находит результаты
- [ ] Статистика считается правильно
- [ ] Экспорт генерирует валидные файлы
- [ ] Unit и integration тесты проходят
- [ ] Swagger документация обновлена (/docs)

---

## Оценка времени

- Репозитории и DTOs: **4-6 часов**
- API endpoints: **4-6 часов**
- Тестирование: **3-4 часа**
- Документация: **1-2 часа**

**Итого: 2-3 дня** (с учетом отладки)

---

## Примечания

1. **Не добавляем сложную логику** - это read-only endpoints
2. **PostgreSQL FTS** - использовать существующий GIN индекс на preprocessed_issues
3. **Экспорт** - для больших выборок использовать StreamingResponse
4. **Errors** - возвращать 404 если issue/cluster не найден, 400 для invalid filters
