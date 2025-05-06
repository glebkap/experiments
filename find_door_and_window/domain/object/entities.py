"""
Entity definitions for doors and windows objects.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple, List


class ObjectType(Enum):
    """Type enumeration for detected objects."""

    DOOR = auto()
    WINDOW = auto()


@dataclass
class Coordinates:
    """Coordinates in the image."""

    x: int
    y: int


@dataclass
class Size:
    """Size of the object."""

    width: int
    height: int


@dataclass
class Object:
    """Base class for detected objects."""

    object_id: str
    object_type: ObjectType
    top_left: Coordinates
    size: Size
    confidence: float = 0.0
    template_type: str = None  # Template name/id that matched this object

    @property
    def bottom_right(self) -> Coordinates:
        """Calculate bottom right coordinates."""
        return Coordinates(
            x=self.top_left.x + self.size.width, y=self.top_left.y + self.size.height
        )

    @property
    def center(self) -> Coordinates:
        """Calculate center coordinates."""
        return Coordinates(
            x=self.top_left.x + self.size.width // 2,
            y=self.top_left.y + self.size.height // 2,
        )

    @property
    def bbox(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Return bounding box coordinates in format ((x1, y1), (x2, y2))."""
        return (
            (self.top_left.x, self.top_left.y),
            (self.top_left.x + self.size.width, self.top_left.y + self.size.height),
        )


@dataclass
class Door(Object):
    """Door object."""

    def __post_init__(self):
        """Initialize with door type."""
        # Only set object_type if not already set
        if not self.object_type:
            self.object_type = ObjectType.DOOR


@dataclass
class Window(Object):
    """Window object."""

    def __post_init__(self):
        """Initialize with window type."""
        # Only set object_type if not already set
        if not self.object_type:
            self.object_type = ObjectType.WINDOW


@dataclass
class DetectedObjects:
    """Container for detected objects."""

    doors: List[Door] = None
    windows: List[Window] = None

    def __post_init__(self):
        """Initialize empty lists if None."""
        self.doors = self.doors or []
        self.windows = self.windows or []

    @property
    def door_count(self) -> int:
        """Return the count of doors."""
        return len(self.doors)

    @property
    def window_count(self) -> int:
        """Return the count of windows."""
        return len(self.windows)

    @property
    def total_count(self) -> int:
        """Return the total count of objects."""
        return self.door_count + self.window_count
