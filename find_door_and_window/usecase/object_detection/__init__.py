"""
Object detection usecase module.
"""

from pathlib import Path
from typing import Tuple, Dict, Any

from domain.object.repository import (
    DoorTemplateRepository,
    WindowTemplateRepository,
    ObjectRepository,
)

from .find_objects import FindObjectsUseCase


def create_find_objects_usecase(
    door_repository: DoorTemplateRepository,
    window_repository: WindowTemplateRepository,
    object_repository: ObjectRepository,
    config: Dict[str, Any] = None,
) -> FindObjectsUseCase:
    """
    Factory function to create a FindObjectsUseCase instance.

    Args:
        door_repository: Repository for door templates
        window_repository: Repository for window templates
        object_repository: Repository for detected objects
        config: Optional configuration dictionary

    Returns:
        Configured FindObjectsUseCase instance
    """
    default_config = {
        # Template matching parameters
        "template_matching_threshold": 0.7,
        "nms_threshold": 0.3,
        # ORB parameters
        "orb_features": 1000,  # Maximum number of features to detect
        "orb_scale_factor": 1.2,  # Pyramid scale factor
        "orb_levels": 8,  # Number of pyramid levels
        "orb_min_matches": 10,  # Minimum number of good matches to consider
        "orb_match_ratio": 0.75,  # Ratio threshold for Lowe's ratio test (KNN matching)
    }

    # Override defaults with provided config
    if config:
        default_config.update(config)

    return FindObjectsUseCase(
        door_repository=door_repository,
        window_repository=window_repository,
        object_repository=object_repository,
        config=default_config,
    )
