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


class FindObjectsUseCase:
    """
    Usecase for finding doors and windows on building plans.

    This implements the algorithm described in the project documentation:
    1. Preprocess the input image
    2. Prepare reference templates
    3. Find objects using ORB feature detection and matching
    4. Find objects using template matching as fallback
    5. Postprocess and refine results
    6. Mark and classify objects
    """

    def __init__(
        self,
        door_repository: DoorTemplateRepository,
        window_repository: WindowTemplateRepository,
        object_repository: ObjectRepository,
        config: Dict[str, Any],
    ):
        """
        Initialize the usecase with required repositories and configuration.

        Args:
            door_repository: Repository for door templates
            window_repository: Repository for window templates
            object_repository: Repository for storing detection results
            config: Configuration dictionary with parameters for algorithms
        """
        self.door_repository = door_repository
        self.window_repository = window_repository
        self.object_repository = object_repository
        self.config = config

    def execute(self, image_path: Path) -> DetectedObjects:
        """
        Execute the object detection process on a building plan image.

        Args:
            image_path: Path to the image file to analyze

        Returns:
            DetectedObjects: Container with detected doors and windows

        Raises:
            ValueError: If image_path is invalid or image cannot be loaded
        """
        # Process the image
        detected_objects = self._process_image(image_path)

        # Store the results
        self.object_repository.save_doors(detected_objects.doors)
        self.object_repository.save_windows(detected_objects.windows)

        return detected_objects

    def _process_image(self, image_path: Path) -> DetectedObjects:
        """
        Process a building plan image to find doors and windows.

        Args:
            image_path: Path to the image file

        Returns:
            DetectedObjects: Container with detected doors and windows
        """
        # Step 1: Load and preprocess the image
        image = self._load_and_preprocess_image(image_path)

        # Step 2: Load and prepare templates
        door_templates = self._load_templates(self.door_repository.get_all_templates())
        window_templates = self._load_templates(
            self.window_repository.get_all_templates()
        )

        # Step 3: Detect objects using ORB feature matching (rotation invariant)
        orb_doors, orb_windows = self._detect_objects_by_orb(
            image, door_templates, window_templates
        )

        # Step 4: As a fallback, use template matching
        template_doors, template_windows = self._detect_objects_by_template_matching(
            image, door_templates, window_templates
        )

        # Step 5: Merge results and remove duplicates
        all_doors = self._merge_and_filter_objects(orb_doors, template_doors)
        all_windows = self._merge_and_filter_objects(orb_windows, template_windows)

        return DetectedObjects(doors=all_doors, windows=all_windows)

    def _detect_objects_by_orb(
        self,
        image_dict: Dict[str, np.ndarray],
        door_templates: List[Dict[str, Any]],
        window_templates: List[Dict[str, Any]],
    ) -> Tuple[List[Door], List[Window]]:
        """
        Detect objects using ORB feature detection and matching.
        This method is invariant to rotation, making it suitable for detecting
        doors and windows regardless of their orientation.

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

        # Initialize ORB detector
        orb = cv2.ORB_create(
            nfeatures=self.config.get("orb_features", 1000),
            scaleFactor=self.config.get("orb_scale_factor", 1.2),
            nlevels=self.config.get("orb_levels", 8),
        )

        # Initialize the feature matcher for KNN matching instead of BFMatcher with crossCheck
        # Using KNN matching with k=2 for Lowe's ratio test
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Detect keypoints and compute descriptors for the input image
        kp_image, des_image = orb.detectAndCompute(gray_image, None)

        # If no keypoints found in the image, return empty results
        if des_image is None or len(des_image) == 0:
            return doors, windows

        # Process each door template individually
        for door_template in door_templates:
            detected_doors = self._process_template_with_orb(
                orb, bf, gray_image, kp_image, des_image, door_template, True
            )
            doors.extend(detected_doors)

        # Process each window template individually
        for window_template in window_templates:
            detected_windows = self._process_template_with_orb(
                orb, bf, gray_image, kp_image, des_image, window_template, False
            )
            windows.extend(detected_windows)

        return doors, windows

    def _process_template_with_orb(
        self,
        orb,
        matcher,
        image: np.ndarray,
        kp_image,
        des_image,
        template: Dict[str, Any],
        is_door: bool,
    ) -> List[Any]:
        """
        Process a single template using ORB feature matching.

        Args:
            orb: ORB detector
            matcher: Feature matcher
            image: Grayscale input image
            kp_image: Keypoints from the input image
            des_image: Descriptors from the input image
            template: Template to process
            is_door: Flag indicating if processing a door (True) or window (False)

        Returns:
            List of detected objects (Door or Window)
        """
        detected_objects = []
        min_matches = self.config.get("orb_min_matches", 10)
        match_ratio = self.config.get("orb_match_ratio", 0.75)
        template_img = template["image"]
        template_type = template["id"]  # Use template ID as object type

        # Detect keypoints and compute descriptors for the template
        kp_template, des_template = orb.detectAndCompute(template_img, None)

        # Skip if no keypoints found in template
        if des_template is None or len(des_template) == 0:
            return detected_objects

        # Use KNN matching instead of simple matching
        # Match with k=2 to apply Lowe's ratio test
        matches = matcher.knnMatch(des_template, des_image, k=2)

        # Apply Lowe's ratio test to filter good matches
        good_matches = []
        for m, n in matches:
            if m.distance < match_ratio * n.distance:
                good_matches.append(m)

        # If we have enough good matches
        if len(good_matches) >= min_matches:
            # Extract matched keypoints
            template_pts = np.float32(
                [kp_template[m.queryIdx].pt for m in good_matches]
            )
            image_pts = np.float32([kp_image[m.trainIdx].pt for m in good_matches])

            # Find homography matrix
            H, mask = cv2.findHomography(template_pts, image_pts, cv2.RANSAC, 5.0)

            if H is not None:
                # Get the corners of the template
                h, w = template_img.shape
                template_corners = np.float32(
                    [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]
                ).reshape(-1, 1, 2)

                # Transform corners to image coordinates
                transformed_corners = cv2.perspectiveTransform(template_corners, H)

                # Get bounding box
                x_values = [pt[0][0] for pt in transformed_corners]
                y_values = [pt[0][1] for pt in transformed_corners]

                x_min, x_max = int(min(x_values)), int(max(x_values))
                y_min, y_max = int(min(y_values)), int(max(y_values))

                width = x_max - x_min
                height = y_max - y_min

                # Filter out unlikely detections based on size
                if width > 0 and height > 0:
                    if is_door:
                        obj = Door(
                            object_id=str(uuid.uuid4()),
                            object_type=None,  # Will be set by __post_init__
                            top_left=Coordinates(x=x_min, y=y_min),
                            size=Size(width=width, height=height),
                            confidence=(
                                len(good_matches) / len(kp_template)
                                if len(kp_template) > 0
                                else 0.5
                            ),
                            template_type=template_type,  # Set the template type from ID
                        )
                    else:
                        obj = Window(
                            object_id=str(uuid.uuid4()),
                            object_type=None,  # Will be set by __post_init__
                            top_left=Coordinates(x=x_min, y=y_min),
                            size=Size(width=width, height=height),
                            confidence=(
                                len(good_matches) / len(kp_template)
                                if len(kp_template) > 0
                                else 0.5
                            ),
                            template_type=template_type,  # Set the template type from ID
                        )

                    detected_objects.append(obj)

        return detected_objects

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

        # Process each door template individually
        for template in door_templates:
            matches = self._template_matching(gray_image, template["image"])
            template_type = template["id"]  # Use template ID as object type
            for match in matches:
                x, y = match["position"]
                width, height = template["size"]
                door = Door(
                    object_id=str(uuid.uuid4()),
                    object_type=None,  # Will be set by __post_init__
                    top_left=Coordinates(x=x, y=y),
                    size=Size(width=width, height=height),
                    confidence=match["confidence"],
                    template_type=template_type,  # Set the template type from ID
                )
                doors.append(door)

        # Process each window template individually
        for template in window_templates:
            matches = self._template_matching(gray_image, template["image"])
            template_type = template["id"]  # Use template ID as object type
            for match in matches:
                x, y = match["position"]
                width, height = template["size"]
                window = Window(
                    object_id=str(uuid.uuid4()),
                    object_type=None,  # Will be set by __post_init__
                    top_left=Coordinates(x=x, y=y),
                    size=Size(width=width, height=height),
                    confidence=match["confidence"],
                    template_type=template_type,  # Set the template type from ID
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
