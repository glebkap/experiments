"""
Repository interfaces for the objects domain.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from .entities import Door, Window


class TemplateRepository(ABC):
    """Abstract repository for object templates."""

    @abstractmethod
    def get_all_templates(self) -> List[Path]:
        """Get all template file paths."""
        pass

    @abstractmethod
    def get_template_by_id(self, template_id: str) -> Optional[Path]:
        """Get a specific template by ID."""
        pass


class DoorTemplateRepository(TemplateRepository):
    """Repository for door templates."""

    pass


class WindowTemplateRepository(TemplateRepository):
    """Repository for window templates."""

    pass


class ObjectRepository(ABC):
    """Abstract repository for detected objects."""

    @abstractmethod
    def save_doors(self, doors: List[Door]) -> None:
        """Save detected doors."""
        pass

    @abstractmethod
    def save_windows(self, windows: List[Window]) -> None:
        """Save detected windows."""
        pass

    @abstractmethod
    def get_doors(self) -> List[Door]:
        """Get all detected doors."""
        pass

    @abstractmethod
    def get_windows(self) -> List[Window]:
        """Get all detected windows."""
        pass
