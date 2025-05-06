## template: entity.py

```python
from datetime import datetime
from pydantic import BaseModel


class EntityId(str):
    """
    Идентификатор сущности
    """
    pass


class Entity(BaseModel):
    id: EntityId

    # Примеры различных аттрибутов доменной сущности
    name: str
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime
    # и т.д.
```

## Notes
1. Сущность может иметь конструктор, методы содержащие бизнес-логику и пр. как репрезентация домена.
2. Сущности могут описываться `pydantic` моделями, если это уместно. В простых случаях достаточно Python dataclass.
