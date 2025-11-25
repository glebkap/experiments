"""Input validation utilities."""

from pathlib import Path


def validate_file_exists(file_path: Path) -> bool:
    """Validate that file exists.

    Args:
        file_path: Path to file

    Returns:
        bool: True if file exists

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Not a file: {file_path}")
    return True


def validate_file_extension(file_path: Path, allowed_extensions: list[str]) -> bool:
    """Validate file extension.

    Args:
        file_path: Path to file
        allowed_extensions: List of allowed extensions (e.g., ['.json', '.jsonl'])

    Returns:
        bool: True if extension is valid

    Raises:
        ValueError: If extension is not allowed
    """
    if file_path.suffix.lower() not in allowed_extensions:
        raise ValueError(
            f"Invalid file extension: {file_path.suffix}. "
            f"Allowed: {', '.join(allowed_extensions)}"
        )
    return True
