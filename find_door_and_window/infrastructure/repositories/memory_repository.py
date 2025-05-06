"""
In-memory repository implementations.
"""

from typing import List, Dict

from domain.object.repository import ObjectRepository
from domain.object.entities import Door, Window


class InMemoryObjectRepository(ObjectRepository):
    """In-memory implementation of object repository."""

    def __init__(self):
        """Initialize empty collections."""
        self._doors: List[Door] = []
        self._windows: List[Window] = []

    def save_doors(self, doors: List[Door]) -> None:
        """Save detected doors."""
        self._doors = doors

    def save_windows(self, windows: List[Window]) -> None:
        """Save detected windows."""
        self._windows = windows

    def get_doors(self) -> List[Door]:
        """Get all detected doors."""
        return self._doors

    def get_windows(self) -> List[Window]:
        """Get all detected windows."""
        return self._windows
