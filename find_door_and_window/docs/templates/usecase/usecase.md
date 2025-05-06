## template: usecase.py

Шаблон usecase.

Имя файла должно соответствовать названию usecase.

```python
from dataclasses import dataclass

# import domain entities
from platform_backend.entity.entity import EntityId, Entity
# import domain exceptions
from platform_backend.entity.exceptions import EntityNotFound
# import domain storage
from platform_backend.entity.storage import IEntityStorage


@dataclass
class GetEntityOutput:
    # result of usecase
    entity: Entity
    # ...


class GetEntity:
    _entity_storage: IEntityStorage

    def __init__(
        self,
        entity_storage: IEntityStorage,
        # other usecase dependencies...
    ):
        self._entity_storage = entity_storage

    async def execute(
        self,
        entity_id: EntityId,
    ) -> GetEntityOutput:
        """
        {add docstring here}
        :param entity_id:
            `platform_backend.domain.EntityId`, идентификатор сущности.
        :return:
            `UsecaseNameOutput`
        :raise platform_backend.entity.exceptions.EntityNotFound:
        """
        e = await self._entity_storage.get_entity(entity_id)

        # some business logic...

        return GetEntityOutput(
            entity=e,
        )
