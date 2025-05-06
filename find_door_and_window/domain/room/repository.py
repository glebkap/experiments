"""
Repository interfaces for the room domain.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from .entities import BuildingPlan


class BuildingPlanRepository(ABC):
    """Abstract repository for building plans."""

    @abstractmethod
    def get_plan_by_id(self, plan_id: str) -> Optional[BuildingPlan]:
        """Get building plan by ID."""
        pass

    @abstractmethod
    def get_all_plans(self) -> List[BuildingPlan]:
        """Get all building plans."""
        pass

    @abstractmethod
    def save_plan(self, plan: BuildingPlan) -> None:
        """Save a building plan."""
        pass
