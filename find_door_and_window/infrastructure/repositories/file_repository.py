"""
File system based repositories.
"""

import os
from pathlib import Path
from typing import List, Optional

from domain.object.repository import DoorTemplateRepository, WindowTemplateRepository
from domain.room.repository import BuildingPlanRepository
from domain.room.entities import BuildingPlan


class FileDoorTemplateRepository(DoorTemplateRepository):
    """File system implementation of door template repository."""

    def __init__(self, templates_dir: Path):
        """Initialize with templates directory."""
        self.templates_dir = templates_dir
        if not templates_dir.exists():
            raise ValueError(f"Templates directory does not exist: {templates_dir}")

    def get_all_templates(self) -> List[Path]:
        """Get all door template file paths."""
        return list(self.templates_dir.glob("*.png")) + list(
            self.templates_dir.glob("*.jpg")
        )

    def get_template_by_id(self, template_id: str) -> Optional[Path]:
        """Get a specific door template by ID."""
        for extension in [".png", ".jpg"]:
            template_path = self.templates_dir / f"{template_id}{extension}"
            if template_path.exists():
                return template_path
        return None


class FileWindowTemplateRepository(WindowTemplateRepository):
    """File system implementation of window template repository."""

    def __init__(self, templates_dir: Path):
        """Initialize with templates directory."""
        self.templates_dir = templates_dir
        if not templates_dir.exists():
            raise ValueError(f"Templates directory does not exist: {templates_dir}")

    def get_all_templates(self) -> List[Path]:
        """Get all window template file paths."""
        return list(self.templates_dir.glob("*.png")) + list(
            self.templates_dir.glob("*.jpg")
        )

    def get_template_by_id(self, template_id: str) -> Optional[Path]:
        """Get a specific window template by ID."""
        for extension in [".png", ".jpg"]:
            template_path = self.templates_dir / f"{template_id}{extension}"
            if template_path.exists():
                return template_path
        return None


class FileBuildingPlanRepository(BuildingPlanRepository):
    """File system implementation of building plan repository."""

    def __init__(self, plans_dir: Path):
        """Initialize with plans directory."""
        self.plans_dir = plans_dir
        if not plans_dir.exists():
            raise ValueError(f"Plans directory does not exist: {plans_dir}")

    def get_plan_by_id(self, plan_id: str) -> Optional[BuildingPlan]:
        """Get building plan by ID."""
        for extension in [".png", ".jpg"]:
            plan_path = self.plans_dir / f"{plan_id}{extension}"
            if plan_path.exists():
                return BuildingPlan(plan_id=plan_id, image_path=plan_path, name=plan_id)
        return None

    def get_all_plans(self) -> List[BuildingPlan]:
        """Get all building plans."""
        plans = []
        for file_path in self.plans_dir.glob("*.png"):
            plan_id = file_path.stem
            plans.append(
                BuildingPlan(plan_id=plan_id, image_path=file_path, name=plan_id)
            )
        for file_path in self.plans_dir.glob("*.jpg"):
            plan_id = file_path.stem
            plans.append(
                BuildingPlan(plan_id=plan_id, image_path=file_path, name=plan_id)
            )
        return plans

    def save_plan(self, plan: BuildingPlan) -> None:
        """Save a building plan."""
        # This method would typically save metadata about the plan
        # The actual image file is assumed to already exist
        pass
