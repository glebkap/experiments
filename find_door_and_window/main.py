#!/usr/bin/env python3
"""
Main application for door and window detection in building plans.
"""
import argparse
import os
from pathlib import Path

from infrastructure.repositories.file_repository import (
    FileDoorTemplateRepository,
    FileWindowTemplateRepository,
)
from infrastructure.repositories.memory_repository import InMemoryObjectRepository
from usecase.object_detection import create_find_objects_usecase
from usecase.visualization import create_visualize_objects_usecase


def setup_repositories(data_dir: Path):
    """Set up repositories for the application."""
    # Create directory paths
    door_templates_dir = data_dir / "doors"
    window_templates_dir = data_dir / "windows"

    # Create repositories
    door_repository = FileDoorTemplateRepository(door_templates_dir)
    window_repository = FileWindowTemplateRepository(window_templates_dir)
    object_repository = InMemoryObjectRepository()

    return door_repository, window_repository, object_repository


def process_image(
    image_path: Path, data_dir: Path, output_dir: Path, config: dict = None
):
    """Process a building plan image to detect and visualize doors and windows."""
    # Setup
    door_repo, window_repo, object_repo = setup_repositories(data_dir)

    # Create usecases
    find_objects_usecase = create_find_objects_usecase(
        door_repository=door_repo,
        window_repository=window_repo,
        object_repository=object_repo,
        config=config,
    )

    visualize_usecase = create_visualize_objects_usecase(
        object_repository=object_repo,
    )

    # Execute object detection
    print(f"Detecting objects in image: {image_path}")
    detected_objects = find_objects_usecase.execute(image_path)
    print(
        f"Found {detected_objects.door_count} doors and {detected_objects.window_count} windows"
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Determine output path
    output_path = output_dir / f"{image_path.stem}_marked{image_path.suffix}"

    # Visualize and save results
    result_path = visualize_usecase.execute(image_path, detected_objects, output_path)
    print(f"Visualization saved to: {result_path}")

    return detected_objects, result_path


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Detect doors and windows in building plans"
    )
    parser.add_argument(
        "--image", required=True, help="Path to the building plan image"
    )
    parser.add_argument("--data-dir", default="./data", help="Path to data directory")
    parser.add_argument(
        "--output-dir", default="./results", help="Path to output directory"
    )
    parser.add_argument(
        "--template-threshold",
        type=float,
        default=0.7,
        help="Threshold for template matching (0.0-1.0)",
    )

    # ORB algorithm parameters
    parser.add_argument(
        "--orb-features",
        type=int,
        default=1000,
        help="Maximum number of features for ORB detector",
    )
    parser.add_argument(
        "--orb-min-matches",
        type=int,
        default=10,
        help="Minimum number of good ORB matches to consider an object",
    )
    parser.add_argument(
        "--orb-match-ratio",
        type=float,
        default=0.75,
        help="Ratio for filtering good ORB matches (0.0-1.0)",
    )

    args = parser.parse_args()

    # Prepare paths
    image_path = Path(args.image)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Prepare configuration
    config = {
        # Template matching parameters
        "template_matching_threshold": args.template_threshold,
        # ORB parameters
        "orb_features": args.orb_features,
        "orb_min_matches": args.orb_min_matches,
        "orb_match_ratio": args.orb_match_ratio,
    }

    # Process the plan
    try:
        process_image(image_path, data_dir, output_dir, config)
    except Exception as e:
        print(f"Error processing image: {e}")


if __name__ == "__main__":
    main()
