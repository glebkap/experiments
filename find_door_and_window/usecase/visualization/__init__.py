"""
Visualization usecase module.
"""

from typing import Dict, Any

from domain.object.repository import ObjectRepository
from domain.room.repository import BuildingPlanRepository

from .visualize_objects import VisualizeObjectsUseCase


def create_visualize_objects_usecase(
    object_repository: ObjectRepository,
    building_plan_repository: BuildingPlanRepository,
    config: Dict[str, Any] = None,
) -> VisualizeObjectsUseCase:
    """
    Factory function to create a VisualizeObjectsUseCase instance.

    Args:
        object_repository: Repository for accessing detected objects
        building_plan_repository: Repository for building plans
        config: Optional configuration dictionary

    Returns:
        Configured VisualizeObjectsUseCase instance
    """
    default_config = {
        "door_color": (0, 0, 255),  # Red in BGR
        "window_color": (255, 0, 0),  # Blue in BGR
        "line_thickness": 2,
        "font_scale": 0.5,
        "text_color": (0, 0, 0),  # Black
        "text_thickness": 1,
    }

    # Override defaults with provided config
    if config:
        default_config.update(config)

    return VisualizeObjectsUseCase(
        object_repository=object_repository,
        building_plan_repository=building_plan_repository,
        config=default_config,
    )
