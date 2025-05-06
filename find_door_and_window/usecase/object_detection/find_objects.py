"""
Usecase implementation for finding doors and windows on building plans.
"""

import uuid
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from domain.object.entities import Door, Window, Coordinates, Size, DetectedObjects
from domain.object.repository import (
    DoorTemplateRepository,
    WindowTemplateRepository,
    ObjectRepository,
)
from domain.room.repository import BuildingPlanRepository
from domain.room.entities import BuildingPlan


class FindObjectsUseCase:
    """
    Usecase for finding doors and windows on building plans.

    This implements the algorithm described in the project documentation:
    1. Preprocess the input image
    2. Prepare reference templates
    3. Find objects using template matching
    4. Find objects using contour analysis
    5. Postprocess and refine results
    6. Mark and classify objects
    """

    def __init__(
        self,
        door_repository: DoorTemplateRepository,
        window_repository: WindowTemplateRepository,
        building_plan_repository: BuildingPlanRepository,
        object_repository: ObjectRepository,
        config: Dict[str, Any],
    ):
        """
        Initialize the usecase with required repositories and configuration.

        Args:
            door_repository: Repository for door templates
            window_repository: Repository for window templates
            building_plan_repository: Repository for building plans
            object_repository: Repository for storing detection results
            config: Configuration dictionary with parameters for algorithms
        """
        self.door_repository = door_repository
        self.window_repository = window_repository
        self.building_plan_repository = building_plan_repository
        self.object_repository = object_repository
        self.config = config

    def execute(self, plan_id: str) -> DetectedObjects:
        """
        Execute the object detection process on a building plan.

        Args:
            plan_id: ID of the building plan to analyze

        Returns:
            DetectedObjects: Container with detected doors and windows

        Raises:
            ValueError: If plan_id is invalid or plan cannot be loaded
        """
        # Step 1: Load the building plan
        plan = self.building_plan_repository.get_plan_by_id(plan_id)
        if not plan:
            raise ValueError(f"Building plan not found: {plan_id}")

        # Step 2: Process the plan image
        detected_objects = self._process_image(plan)

        # Step 3: Store the results
        self.object_repository.save_doors(detected_objects.doors)
        self.object_repository.save_windows(detected_objects.windows)

        return detected_objects

    def _process_image(self, plan: BuildingPlan) -> DetectedObjects:
        """
        Process a building plan image to find doors and windows.

        Args:
            plan: Building plan to analyze

        Returns:
            DetectedObjects: Container with detected doors and windows
        """
        # Step 1: Load and preprocess the image
        image = self._load_and_preprocess_image(plan.image_path)

        # Step 2: Load and prepare templates
        door_templates = self._load_templates(self.door_repository.get_all_templates())
        window_templates = self._load_templates(
            self.window_repository.get_all_templates()
        )

        # Step 3: Detect objects using template matching
        template_doors, template_windows = self._detect_objects_by_template_matching(
            image, door_templates, window_templates
        )

        # Step 4: Detect objects using contour analysis
        contour_doors, contour_windows = self._detect_objects_by_contour_analysis(image)

        # Step 5: Merge results and remove duplicates
        all_doors = self._merge_and_filter_objects(template_doors, contour_doors)
        all_windows = self._merge_and_filter_objects(template_windows, contour_windows)

        return DetectedObjects(doors=all_doors, windows=all_windows)

    def _load_and_preprocess_image(self, image_path: Path) -> Dict[str, np.ndarray]:
        """
        Load and preprocess an image for object detection.

        Args:
            image_path: Path to the image file

        Returns:
            Dict containing original and preprocessed image versions
        """
        # Load the image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply adaptive thresholding for binarization
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Apply morphological operations to clean up the binary image
        kernel = np.ones((3, 3), np.uint8)
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return {
            "original": image,
            "gray": gray,
            "blurred": blurred,
            "binary": binary,
            "morphed": morphed,
        }

    def _load_templates(self, template_paths: List[Path]) -> List[Dict[str, Any]]:
        """
        Load and preprocess template images.

        Args:
            template_paths: List of paths to template images

        Returns:
            List of dictionaries containing template images and metadata
        """
        templates = []

        for path in template_paths:
            template_image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if template_image is None:
                continue

            # Store template with its metadata
            templates.append(
                {
                    "id": path.stem,
                    "image": template_image,
                    "size": (template_image.shape[1], template_image.shape[0]),
                }
            )

        return templates

    def _detect_objects_by_template_matching(
        self,
        image_dict: Dict[str, np.ndarray],
        door_templates: List[Dict[str, Any]],
        window_templates: List[Dict[str, Any]],
    ) -> Tuple[List[Door], List[Window]]:
        """
        Detect objects using template matching.

        Args:
            image_dict: Dictionary with preprocessed images
            door_templates: List of door templates
            window_templates: List of window templates

        Returns:
            Tuple of (door list, window list)
        """
        gray_image = image_dict["gray"]
        doors = []
        windows = []

        # Process door templates
        for template in door_templates:
            matches = self._template_matching(gray_image, template["image"])
            for match in matches:
                x, y = match["position"]
                width, height = template["size"]
                door = Door(
                    object_id=str(uuid.uuid4()),
                    object_type=None,  # Will be set by __post_init__
                    top_left=Coordinates(x=x, y=y),
                    size=Size(width=width, height=height),
                    confidence=match["confidence"],
                )
                doors.append(door)

        # Process window templates
        for template in window_templates:
            matches = self._template_matching(gray_image, template["image"])
            for match in matches:
                x, y = match["position"]
                width, height = template["size"]
                window = Window(
                    object_id=str(uuid.uuid4()),
                    object_type=None,  # Will be set by __post_init__
                    top_left=Coordinates(x=x, y=y),
                    size=Size(width=width, height=height),
                    confidence=match["confidence"],
                )
                windows.append(window)

        return doors, windows

    def _template_matching(
        self, image: np.ndarray, template: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Perform template matching and return matches.

        Args:
            image: Image to search in
            template: Template to search for

        Returns:
            List of matches with positions and confidence
        """
        # Get template dimensions
        h, w = template.shape

        # Apply template matching
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

        # Find positions where the match exceeds the threshold
        threshold = self.config.get("template_matching_threshold", 0.7)
        locations = np.where(result >= threshold)

        # Convert to list of match dictionaries
        matches = []
        for pt in zip(*locations[::-1]):  # Reverse to get (x, y)
            matches.append({"position": pt, "confidence": result[pt[1], pt[0]]})

        # Apply non-maximum suppression to remove overlapping detections
        return self._non_maximum_suppression(matches, w, h)

    def _non_maximum_suppression(
        self, matches: List[Dict[str, Any]], width: int, height: int
    ) -> List[Dict[str, Any]]:
        """
        Apply non-maximum suppression to remove overlapping detections.

        Args:
            matches: List of match dictionaries
            width: Width of the detected object
            height: Height of the detected object

        Returns:
            Filtered list of matches
        """
        if not matches:
            return []

        # Sort matches by confidence (descending)
        matches = sorted(matches, key=lambda x: x["confidence"], reverse=True)

        # Initialize list of picked matches
        picked = []

        # Extract coordinates
        x1_values = [m["position"][0] for m in matches]
        y1_values = [m["position"][1] for m in matches]
        x2_values = [m["position"][0] + width for m in matches]
        y2_values = [m["position"][1] + height for m in matches]

        # Compute areas
        areas = [width * height] * len(matches)

        # Process matches
        indices = list(range(len(matches)))

        while indices:
            # Pick the match with highest confidence
            current = indices[0]
            picked.append(matches[current])

            # Find remaining indices
            indices_to_keep = []

            for idx in indices[1:]:
                # Calculate intersection
                xx1 = max(x1_values[current], x1_values[idx])
                yy1 = max(y1_values[current], y1_values[idx])
                xx2 = min(x2_values[current], x2_values[idx])
                yy2 = min(y2_values[current], y2_values[idx])

                # Check if there is an intersection
                width_intersection = max(0, xx2 - xx1)
                height_intersection = max(0, yy2 - yy1)

                if width_intersection == 0 or height_intersection == 0:
                    indices_to_keep.append(idx)
                    continue

                # Calculate intersection area
                intersection_area = width_intersection * height_intersection

                # Calculate IoU
                iou = intersection_area / (
                    areas[current] + areas[idx] - intersection_area
                )

                # If IoU is less than threshold, keep the match
                if iou < self.config.get("nms_threshold", 0.3):
                    indices_to_keep.append(idx)

            indices = indices_to_keep

        return picked

    def _detect_objects_by_contour_analysis(
        self, image_dict: Dict[str, np.ndarray]
    ) -> Tuple[List[Door], List[Window]]:
        """
        Detect objects by analyzing contours in the image.

        Args:
            image_dict: Dictionary with preprocessed images

        Returns:
            Tuple of (door list, window list)
        """
        binary_image = image_dict["morphed"]
        original_image = image_dict["original"]

        # Find contours
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        doors = []
        windows = []

        min_area = self.config.get("min_contour_area", 100)
        max_area = self.config.get("max_contour_area", 10000)

        for contour in contours:
            # Calculate contour area and filter by size
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Analyze shape to classify as door or window
            aspect_ratio = float(w) / h if h > 0 else 0

            # Create object with appropriate type
            if self._is_likely_door(contour, aspect_ratio):
                door = Door(
                    object_id=str(uuid.uuid4()),
                    object_type=None,  # Will be set by __post_init__
                    top_left=Coordinates(x=x, y=y),
                    size=Size(width=w, height=h),
                    confidence=0.7,  # Default confidence for contour method
                )
                doors.append(door)
            elif self._is_likely_window(contour, aspect_ratio):
                window = Window(
                    object_id=str(uuid.uuid4()),
                    object_type=None,  # Will be set by __post_init__
                    top_left=Coordinates(x=x, y=y),
                    size=Size(width=w, height=h),
                    confidence=0.7,  # Default confidence for contour method
                )
                windows.append(window)

        return doors, windows

    def _is_likely_door(self, contour: np.ndarray, aspect_ratio: float) -> bool:
        """
        Determine if a contour is likely to be a door based on shape analysis.

        Args:
            contour: Contour to analyze
            aspect_ratio: Width to height ratio

        Returns:
            True if the contour is likely a door, False otherwise
        """
        # Doors typically have aspect ratios around 0.5 (taller than wide)
        # and are relatively rectangular
        if 0.2 <= aspect_ratio <= 0.8:
            # Check rectangularity
            rect_area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            bounding_rect_area = w * h

            if bounding_rect_area > 0:
                rectangularity = rect_area / bounding_rect_area

                # Doors are typically rectangular (high rectangularity)
                if rectangularity > 0.7:
                    return True

        return False

    def _is_likely_window(self, contour: np.ndarray, aspect_ratio: float) -> bool:
        """
        Determine if a contour is likely to be a window based on shape analysis.

        Args:
            contour: Contour to analyze
            aspect_ratio: Width to height ratio

        Returns:
            True if the contour is likely a window, False otherwise
        """
        # Windows typically have aspect ratios closer to 1 or wider
        # and are also rectangular
        if 0.8 <= aspect_ratio <= 2.5:
            # Check rectangularity
            rect_area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            bounding_rect_area = w * h

            if bounding_rect_area > 0:
                rectangularity = rect_area / bounding_rect_area

                # Windows are typically rectangular (high rectangularity)
                if rectangularity > 0.7:
                    return True

        return False

    def _merge_and_filter_objects(
        self, objects1: List[Any], objects2: List[Any]
    ) -> List[Any]:
        """
        Merge and filter objects from different detection methods.

        Args:
            objects1: First list of objects
            objects2: Second list of objects

        Returns:
            Merged and filtered list of objects
        """
        # Combine all objects
        all_objects = objects1 + objects2

        # Sort by confidence
        all_objects.sort(key=lambda obj: obj.confidence, reverse=True)

        # Filter out overlapping objects (similar to NMS)
        filtered_objects = []

        for obj in all_objects:
            # If we haven't added any objects yet, add the first one
            if not filtered_objects:
                filtered_objects.append(obj)
                continue

            # Check for overlaps with existing objects
            is_overlapping = False
            for existing_obj in filtered_objects:
                if self._calculate_iou(obj, existing_obj) > self.config.get(
                    "nms_threshold", 0.3
                ):
                    is_overlapping = True
                    break

            # If no significant overlap, add the object
            if not is_overlapping:
                filtered_objects.append(obj)

        return filtered_objects

    def _calculate_iou(self, obj1: Any, obj2: Any) -> float:
        """
        Calculate Intersection over Union (IoU) between two objects.

        Args:
            obj1: First object
            obj2: Second object

        Returns:
            IoU value (0 to 1)
        """
        # Extract bounding box coordinates
        x1_1, y1_1 = obj1.top_left.x, obj1.top_left.y
        x2_1, y2_1 = x1_1 + obj1.size.width, y1_1 + obj1.size.height

        x1_2, y1_2 = obj2.top_left.x, obj2.top_left.y
        x2_2, y2_2 = x1_2 + obj2.size.width, y1_2 + obj2.size.height

        # Calculate intersection coordinates
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        # Check if there is an intersection
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        # Calculate areas
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = obj1.size.width * obj1.size.height
        area2 = obj2.size.width * obj2.size.height

        # Calculate IoU
        return intersection_area / (area1 + area2 - intersection_area)
