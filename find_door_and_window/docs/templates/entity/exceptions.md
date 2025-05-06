## template: exceptions.md

```python
class EntityBaseException(Exception):
    pass


class EntityNotFound(EntityBaseException):
    """
    Сущность не найдена.
    """
    pass


class EntityConflict(EntityBaseException):
    """
    Такая сущность уже существует.
    """
    pass
```

## Notes
1. Доменные исключения должны наследоваться от базового класса. Например, `EntityBaseException`. Это позволяет перехватывать все исключения сущности.
2. Каждая сущность в рамках доменного пакета может обладать своим набором исключений.
3. Исключения могут включать дополнительный контекст при необходимости. Например, в `EntityConflict` может содержать причина конфликта.
