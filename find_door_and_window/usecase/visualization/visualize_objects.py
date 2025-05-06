"""
Usecase implementation for visualizing detected doors and windows.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from domain.object.entities import Door, Window, DetectedObjects
from domain.object.repository import ObjectRepository


class VisualizeObjectsUseCase:
    """
    Usecase for visualizing detected doors and windows on building plans.
    """

    def __init__(
        self,
        object_repository: ObjectRepository,
        config: Dict[str, Any],
    ):
        """
        Initialize the usecase with required repositories and configuration.

        Args:
            object_repository: Repository for accessing detected objects
            config: Configuration dictionary with visualization parameters
        """
        self.object_repository = object_repository
        self.config = config

    def execute(
        self,
        image_path: Path,
        detected_objects: DetectedObjects,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Execute the visualization process.

        Args:
            image_path: Path to the input image
            detected_objects: Objects detected on the image
            output_path: Optional path to save the output image

        Returns:
            Path to the output visualized image

        Raises:
            ValueError: If image_path is invalid or image cannot be loaded
        """
        # Step 1: Create visualization
        result_image = self._create_visualization(image_path, detected_objects)

        # Step 2: Save the visualization if output path provided
        if output_path is None:
            # Create output path based on input path
            output_path = (
                image_path.parent / f"{image_path.stem}_marked{image_path.suffix}"
            )

        # Ensure the directory exists
        os.makedirs(output_path.parent, exist_ok=True)

        # Save the image
        cv2.imwrite(str(output_path), result_image)

        return output_path

    def _create_visualization(
        self, image_path: Path, objects: DetectedObjects
    ) -> np.ndarray:
        """
        Create a visualization of the detected objects on the building plan.

        Args:
            image_path: Path to the input image
            objects: Detected objects to visualize

        Returns:
            Visualization image with marked objects
        """
        # Load the original image
        original_image = cv2.imread(str(image_path))
        if original_image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Create a copy for visualization
        visualization = original_image.copy()

        # Draw doors
        for i, door in enumerate(objects.doors):
            self._draw_object(
                visualization,
                door,
                self.config.get("door_color", (0, 0, 255)),  # Red in BGR
                f"Door {i+1}",
            )

        # Draw windows
        for i, window in enumerate(objects.windows):
            self._draw_object(
                visualization,
                window,
                self.config.get("window_color", (255, 0, 0)),  # Blue in BGR
                f"Window {i+1}",
            )

        # Add statistics to the image
        self._add_statistics(visualization, objects)

        return visualization

    def _draw_object(
        self,
        image: np.ndarray,
        obj: Door or Window,
        color: Tuple[int, int, int],
        label: str,
    ) -> None:
        """
        Draw an object on the image.

        Args:
            image: Image to draw on
            obj: Object to draw
            color: BGR color tuple
            label: Label text
        """
        # Extract object coordinates
        x1, y1 = obj.top_left.x, obj.top_left.y
        x2, y2 = x1 + obj.size.width, y1 + obj.size.height

        # Draw rectangle
        cv2.rectangle(
            image, (x1, y1), (x2, y2), color, self.config.get("line_thickness", 2)
        )

        # Add template type to label if available
        if obj.template_type:
            label = f"{label} ({obj.template_type})"

        # Add label
        cv2.putText(
            image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.get("font_scale", 0.5),
            self.config.get("text_color", (0, 0, 0)),
            self.config.get("text_thickness", 1),
        )

    def _add_statistics(self, image: np.ndarray, objects: DetectedObjects) -> None:
        """
        Add statistics text to the image.

        Args:
            image: Image to add statistics to
            objects: Detected objects
        """
        # Prepare statistics text
        stats_text = [
            f"Total doors: {objects.door_count}",
            f"Total windows: {objects.window_count}",
            f"Total objects: {objects.total_count}",
        ]

        # Define position for text (top-right corner)
        y_offset = 30
        for text in stats_text:
            cv2.putText(
                image,
                text,
                (image.shape[1] - 250, y_offset),  # Increased x offset for more space
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.get("font_scale", 0.5),
                self.config.get("text_color", (0, 0, 0)),
                self.config.get("text_thickness", 1),
            )
            y_offset += 20
