## template: __init__.py

```python
from platform_backend.entities.get_entity import GetEntity


class Usecases:
    get_entity: GetEntity
    # other possible usecases of the module

    def __init__(
        self,
        get_entity_uc: GetEntity,
        # other possible usecases
    ):
        self.get_entity = get_entity_uc
        # assign usecases
```
