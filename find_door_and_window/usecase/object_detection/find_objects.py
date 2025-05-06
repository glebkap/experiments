"""
Usecase implementation for finding doors and windows on building plans.
"""

import uuid
import cv2
import numpy as np
import os
import logging
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

        # Configure logging
        self.debug_mode = config.get("debug", False)
        if self.debug_mode:
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
        else:
            logging.basicConfig(level=logging.INFO)

        self.logger = logging.getLogger("FindObjectsUseCase")

        # Create debug directory if needed
        if self.debug_mode:
            self.debug_dir = Path("results/debug")
            os.makedirs(self.debug_dir, exist_ok=True)
            self.logger.debug(
                "Debug mode enabled. Images will be saved to %s", self.debug_dir
            )

    def _save_debug_image(
        self, name: str, image: np.ndarray, image_path: Optional[Path] = None
    ):
        """
        Save an image for debugging purposes.

        Args:
            name: Name to identify the debug image
            image: Image to save
            image_path: Original image path (to extract filename)
        """
        if not self.debug_mode:
            return

        if image_path:
            filename = f"{image_path.stem}_{name}{image_path.suffix}"
        else:
            filename = f"debug_{name}.png"

        output_path = self.debug_dir / filename
        cv2.imwrite(str(output_path), image)
        self.logger.debug(f"Saved debug image: {output_path}")

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
        if self.debug_mode:
            self.logger.debug(f"Processing image: {image_path}")

        # Process the image
        detected_objects = self._process_image(image_path)

        # Store the results
        self.object_repository.save_doors(detected_objects.doors)
        self.object_repository.save_windows(detected_objects.windows)

        if self.debug_mode:
            self.logger.debug(
                f"Detection complete. Found {detected_objects.door_count} doors and {detected_objects.window_count} windows"
            )

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

        # Save preprocessed images for debugging
        if self.debug_mode:
            self._save_debug_image("1_gray", image["gray"], image_path)
            self.logger.debug("Preprocessing complete")

        # Step 2: Load and prepare templates
        door_templates = self._load_templates(self.door_repository.get_all_templates())
        window_templates = self._load_templates(
            self.window_repository.get_all_templates()
        )

        if self.debug_mode:
            self.logger.debug(
                f"Loaded {len(door_templates)} door templates and {len(window_templates)} window templates"
            )

        # Step 3: Detect objects using ORB feature matching (rotation invariant)
        orb_doors, orb_windows = self._detect_objects_by_orb(
            image, door_templates, window_templates
        )

        if self.debug_mode:
            self.logger.debug(
                f"ORB detection complete. Found {len(orb_doors)} doors and {len(orb_windows)} windows"
            )
            # Create visualization of ORB results
            orig_img = cv2.imread(str(image_path))
            orb_viz = orig_img.copy()

            # Draw ORB detected doors
            for i, door in enumerate(orb_doors):
                cv2.rectangle(
                    orb_viz,
                    (door.top_left.x, door.top_left.y),
                    (
                        door.top_left.x + door.size.width,
                        door.top_left.y + door.size.height,
                    ),
                    (0, 0, 255),  # Red in BGR
                    2,
                )

            # Draw ORB detected windows
            for i, window in enumerate(orb_windows):
                cv2.rectangle(
                    orb_viz,
                    (window.top_left.x, window.top_left.y),
                    (
                        window.top_left.x + window.size.width,
                        window.top_left.y + window.size.height,
                    ),
                    (255, 0, 0),  # Blue in BGR
                    2,
                )

            self._save_debug_image("2_orb_detection", orb_viz, image_path)

        # Step 4: As a fallback, use template matching
        template_doors, template_windows = self._detect_objects_by_template_matching(
            image, door_templates, window_templates
        )

        if self.debug_mode:
            self.logger.debug(
                f"Template matching complete. Found {len(template_doors)} doors and {len(template_windows)} windows"
            )
            # Create visualization of template matching results
            orig_img = cv2.imread(str(image_path))
            template_viz = orig_img.copy()

            # Draw template detected doors
            for i, door in enumerate(template_doors):
                cv2.rectangle(
                    template_viz,
                    (door.top_left.x, door.top_left.y),
                    (
                        door.top_left.x + door.size.width,
                        door.top_left.y + door.size.height,
                    ),
                    (0, 255, 0),  # Green in BGR
                    2,
                )

            # Draw template detected windows
            for i, window in enumerate(template_windows):
                cv2.rectangle(
                    template_viz,
                    (window.top_left.x, window.top_left.y),
                    (
                        window.top_left.x + window.size.width,
                        window.top_left.y + window.size.height,
                    ),
                    (255, 255, 0),  # Cyan in BGR
                    2,
                )

            self._save_debug_image("3_template_matching", template_viz, image_path)

        # Step 5: Merge results and remove duplicates
        all_doors = self._merge_and_filter_objects(orb_doors, template_doors)
        all_windows = self._merge_and_filter_objects(orb_windows, template_windows)

        if self.debug_mode:
            self.logger.debug(
                f"After merging and filtering: {len(all_doors)} doors and {len(all_windows)} windows"
            )
            # Create visualization of final results
            orig_img = cv2.imread(str(image_path))
            final_viz = orig_img.copy()

            # Draw final doors
            for i, door in enumerate(all_doors):
                cv2.rectangle(
                    final_viz,
                    (door.top_left.x, door.top_left.y),
                    (
                        door.top_left.x + door.size.width,
                        door.top_left.y + door.size.height,
                    ),
                    (0, 0, 255),  # Red in BGR
                    2,
                )

            # Draw final windows
            for i, window in enumerate(all_windows):
                cv2.rectangle(
                    final_viz,
                    (window.top_left.x, window.top_left.y),
                    (
                        window.top_left.x + window.size.width,
                        window.top_left.y + window.size.height,
                    ),
                    (255, 0, 0),  # Blue in BGR
                    2,
                )

            self._save_debug_image("4_final_detection", final_viz, image_path)

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
                            object_type="door",
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
                            object_type="window",
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
        Load and preprocess the building plan image.

        Args:
            image_path: Path to the input image

        Returns:
            Dictionary with preprocessed images (original, gray)

        Raises:
            ValueError: If image cannot be loaded
        """
        if self.debug_mode:
            self.logger.debug(f"Loading image: {image_path}")

        # Load the image
        original = cv2.imread(str(image_path))
        if original is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        if self.debug_mode:
            self.logger.debug("Converted image to grayscale")

        # Return preprocessed images
        return {
            "original": original,
            "gray": gray,
        }

    def _load_templates(self, template_paths: List[Path]) -> List[Dict[str, Any]]:
        """
        Load and prepare templates for detection.

        Args:
            template_paths: List of paths to template images

        Returns:
            List of loaded templates with metadata
        """
        templates = []

        if self.debug_mode:
            self.logger.debug(f"Loading {len(template_paths)} templates")

        for i, path in enumerate(template_paths):
            # Extract template type from filename
            template_type = path.stem

            # Load image
            template_img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if template_img is None:
                if self.debug_mode:
                    self.logger.warning(f"Failed to load template: {path}")
                continue

            if self.debug_mode:
                self.logger.debug(f"Loaded template {template_type} from {path}")
                self._save_debug_image(f"template_{i}_{template_type}", template_img)

            # Store template with metadata
            templates.append(
                {
                    "id": template_type,
                    "image": template_img,
                    "path": path,
                    "width": template_img.shape[1],
                    "height": template_img.shape[0],
                }
            )

        if self.debug_mode:
            self.logger.debug(f"Successfully loaded {len(templates)} templates")

        return templates

    def _detect_objects_by_template_matching(
        self,
        image_dict: Dict[str, np.ndarray],
        door_templates: List[Dict[str, Any]],
        window_templates: List[Dict[str, Any]],
    ) -> Tuple[List[Door], List[Window]]:
        """
        Detect objects using template matching algorithm.

        Args:
            image_dict: Dictionary with preprocessed images
            door_templates: List of door templates
            window_templates: List of window templates

        Returns:
            Tuple of (door list, window list)
        """
        if self.debug_mode:
            self.logger.debug("Starting template matching detection")

        gray_image = image_dict["gray"]
        doors = []
        windows = []

        threshold = self.config.get("template_matching_threshold", 0.7)

        if self.debug_mode:
            self.logger.debug(f"Using template matching threshold: {threshold}")

        # Process door templates
        for door_template in door_templates:
            if self.debug_mode:
                self.logger.debug(f"Matching door template: {door_template['id']}")

            template_img = door_template["image"]
            template_type = door_template["id"]

            # Perform template matching
            matches = self._template_matching(gray_image, template_img)

            if self.debug_mode:
                self.logger.debug(
                    f"Found {len(matches)} potential matches before filtering"
                )

            # Convert matches to Door objects
            for match in matches:
                confidence = match["confidence"]

                # Filter by confidence threshold
                if confidence < threshold:
                    continue

                # Create a Door object
                x, y = match["location"]
                w, h = template_img.shape[1], template_img.shape[0]

                door = Door(
                    object_id=str(uuid.uuid4()),
                    object_type="door",
                    top_left=Coordinates(x=x, y=y),
                    size=Size(width=w, height=h),
                    confidence=confidence,
                    template_type=template_type,
                )
                doors.append(door)

            if self.debug_mode:
                self.logger.debug(
                    f"Added {len(doors)} doors after filtering by confidence"
                )

        # Process window templates
        for window_template in window_templates:
            if self.debug_mode:
                self.logger.debug(f"Matching window template: {window_template['id']}")

            template_img = window_template["image"]
            template_type = window_template["id"]

            # Perform template matching
            matches = self._template_matching(gray_image, template_img)

            if self.debug_mode:
                self.logger.debug(
                    f"Found {len(matches)} potential matches before filtering"
                )

            # Convert matches to Window objects
            for match in matches:
                confidence = match["confidence"]

                # Filter by confidence threshold
                if confidence < threshold:
                    continue

                # Create a Window object
                x, y = match["location"]
                w, h = template_img.shape[1], template_img.shape[0]

                window = Window(
                    object_id=str(uuid.uuid4()),
                    object_type="window",
                    top_left=Coordinates(x=x, y=y),
                    size=Size(width=w, height=h),
                    confidence=confidence,
                    template_type=template_type,
                )
                windows.append(window)

            if self.debug_mode:
                self.logger.debug(
                    f"Added {len(windows)} windows after filtering by confidence"
                )

        return doors, windows

    def _template_matching(
        self, image: np.ndarray, template: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Perform template matching on an image.

        Args:
            image: Image to search in
            template: Template to search for

        Returns:
            List of match dictionaries with position and confidence
        """
        if self.debug_mode:
            self.logger.debug(
                f"Performing template matching with template size: {template.shape}"
            )

        # Get image dimensions
        img_height, img_width = image.shape
        template_height, template_width = template.shape

        # If template is larger than image, skip
        if template_height > img_height or template_width > img_width:
            if self.debug_mode:
                self.logger.debug("Template is larger than image, skipping")
            return []

        # Perform template matching
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

        if self.debug_mode:
            # Save template matching result as heatmap
            result_normalized = cv2.normalize(
                result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
            )
            result_heatmap = cv2.applyColorMap(result_normalized, cv2.COLORMAP_JET)
            self._save_debug_image(
                f"template_matching_heatmap_{template.shape[0]}x{template.shape[1]}",
                result_heatmap,
            )

        # Find potential matches
        potential_matches = []
        loc = np.where(result >= self.config.get("template_matching_threshold", 0.7))

        for y, x in zip(*loc):
            potential_matches.append({"location": (x, y), "confidence": result[y, x]})

        if self.debug_mode:
            self.logger.debug(
                f"Found {len(potential_matches)} potential matches above threshold"
            )

        # Apply non-maximum suppression to remove overlapping detections
        final_matches = self._non_maximum_suppression(
            potential_matches, template_width, template_height
        )

        if self.debug_mode:
            self.logger.debug(
                f"After non-maximum suppression: {len(final_matches)} matches"
            )

        return final_matches

    def _non_maximum_suppression(
        self, matches: List[Dict[str, Any]], width: int, height: int
    ) -> List[Dict[str, Any]]:
        """
        Apply non-maximum suppression to remove overlapping detections.

        Args:
            matches: List of match dictionaries
            width: Width of the template
            height: Height of the template

        Returns:
            Filtered list of match dictionaries
        """
        if not matches:
            return []

        if self.debug_mode:
            self.logger.debug(
                f"Applying non-maximum suppression to {len(matches)} matches"
            )

        # If only one match, return it directly
        if len(matches) == 1:
            if self.debug_mode:
                self.logger.debug("Only one match found, skipping NMS")
            return matches

        # Sort matches by confidence (highest first)
        matches = sorted(matches, key=lambda x: x["confidence"], reverse=True)

        # Prepare pick list and suppressed flags
        pick = []
        suppressed = [False] * len(matches)

        # Threshold for suppression
        nms_threshold = self.config.get("nms_threshold", 0.3)

        # For each match
        for i in range(len(matches)):
            # Skip if already suppressed
            if suppressed[i]:
                continue

            # Add to picked matches
            pick.append(i)

            # Get coordinates of current match
            x1_i = matches[i]["location"][0]
            y1_i = matches[i]["location"][1]
            x2_i = x1_i + width
            y2_i = y1_i + height
            area_i = width * height

            # Compare with all other matches
            for j in range(i + 1, len(matches)):
                # Skip if already suppressed
                if suppressed[j]:
                    continue

                # Get coordinates of compared match
                x1_j = matches[j]["location"][0]
                y1_j = matches[j]["location"][1]
                x2_j = x1_j + width
                y2_j = y1_j + height
                area_j = width * height

                # Calculate intersection
                xx1 = max(x1_i, x1_j)
                yy1 = max(y1_i, y1_j)
                xx2 = min(x2_i, x2_j)
                yy2 = min(y2_i, y2_j)

                # Check if there's an intersection
                w = max(0, xx2 - xx1 + 1)
                h = max(0, yy2 - yy1 + 1)

                if w > 0 and h > 0:
                    # Calculate intersection area
                    intersection = w * h

                    # Calculate IoU
                    iou = intersection / float(area_i + area_j - intersection)

                    # Suppress if IoU is above threshold
                    if iou > nms_threshold:
                        suppressed[j] = True

        # Return filtered matches
        filtered_matches = [matches[i] for i in pick]

        if self.debug_mode:
            self.logger.debug(
                f"Non-maximum suppression kept {len(filtered_matches)} of {len(matches)} matches"
            )

        return filtered_matches

    def _merge_and_filter_objects(
        self, objects1: List[Any], objects2: List[Any]
    ) -> List[Any]:
        """
        Merge two sets of objects and remove duplicates.

        Args:
            objects1: First list of objects
            objects2: Second list of objects

        Returns:
            Merged list with duplicates removed
        """
        if not objects1 and not objects2:
            return []

        if self.debug_mode:
            self.logger.debug(
                f"Merging object lists: {len(objects1)} and {len(objects2)} objects"
            )

        # Merge the two lists
        all_objects = objects1.copy()

        # Keep track of which objects from list 2 are duplicates
        duplicate_indices = []

        # Add objects from list 2 that don't overlap with list 1
        for i, obj2 in enumerate(objects2):
            is_duplicate = False

            for obj1 in objects1:
                iou = self._calculate_iou(obj1, obj2)

                if iou > self.config.get("duplicate_iou_threshold", 0.3):
                    is_duplicate = True
                    break

            if not is_duplicate:
                all_objects.append(obj2)
            else:
                duplicate_indices.append(i)

        if self.debug_mode:
            self.logger.debug(
                f"Found {len(duplicate_indices)} duplicates between the two lists"
            )
            self.logger.debug(f"Final merged list contains {len(all_objects)} objects")

        return all_objects

    def _calculate_iou(self, obj1: Any, obj2: Any) -> float:
        """
        Calculate Intersection over Union between two objects.

        Args:
            obj1: First object
            obj2: Second object

        Returns:
            IoU value between 0 and 1
        """
        # Extract coordinates
        x1_1 = obj1.top_left.x
        y1_1 = obj1.top_left.y
        x2_1 = x1_1 + obj1.size.width
        y2_1 = y1_1 + obj1.size.height

        x1_2 = obj2.top_left.x
        y1_2 = obj2.top_left.y
        x2_2 = x1_2 + obj2.size.width
        y2_2 = y1_2 + obj2.size.height

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        # Check if there is an intersection
        if x1_i >= x2_i or y1_i >= y2_i:
            return 0.0

        # Calculate areas
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = obj1.size.width * obj1.size.height
        area2 = obj2.size.width * obj2.size.height

        # Calculate IoU
        iou = intersection_area / (area1 + area2 - intersection_area)

        return iou
