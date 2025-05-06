"""
Entity definitions for room domain.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class BuildingPlan:
    """Building plan representation."""

    plan_id: str
    image_path: Path
    name: Optional[str] = None
    description: Optional[str] = None
