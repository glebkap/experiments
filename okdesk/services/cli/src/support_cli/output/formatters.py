"""Output formatters for various data types."""

import json
from datetime import datetime
from typing import Any

from .styles import STATUS_COLORS, get_status_style


def format_datetime(dt: datetime | None) -> str:
    """Format datetime for display.

    Args:
        dt: Datetime object

    Returns:
        str: Formatted datetime string
    """
    if dt is None:
        return "—"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_date(dt: datetime | None) -> str:
    """Format date for display.

    Args:
        dt: Datetime object

    Returns:
        str: Formatted date string
    """
    if dt is None:
        return "—"
    return dt.strftime("%Y-%m-%d")


def format_status(status: str) -> str:
    """Format status with color.

    Args:
        status: Status string

    Returns:
        str: Formatted status with Rich markup
    """
    color = STATUS_COLORS.get(status.lower(), "white")
    return f"[{color}]{status}[/{color}]"


def format_json_pretty(data: Any) -> str:
    """Format JSON data with indentation.

    Args:
        data: Data to format as JSON

    Returns:
        str: Pretty-printed JSON string
    """
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)


def format_file_size(bytes_size: int | None) -> str:
    """Format file size in human-readable format.

    Args:
        bytes_size: Size in bytes

    Returns:
        str: Formatted file size (e.g., "1.5 MB")
    """
    if bytes_size is None:
        return "—"

    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


def format_duration(seconds: float | None) -> str:
    """Format duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        str: Formatted duration (e.g., "1m 30s")
    """
    if seconds is None:
        return "—"

    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60

    if minutes < 60:
        return f"{minutes}m {remaining_seconds:.0f}s"

    hours = minutes // 60
    remaining_minutes = minutes % 60
    return f"{hours}h {remaining_minutes}m"


def format_percentage(value: float | None, decimals: int = 1) -> str:
    """Format percentage.

    Args:
        value: Value between 0 and 1
        decimals: Number of decimal places

    Returns:
        str: Formatted percentage (e.g., "75.5%")
    """
    if value is None:
        return "—"
    return f"{value * 100:.{decimals}f}%"


def format_similarity(similarity: float | None) -> str:
    """Format similarity score with color.

    Args:
        similarity: Similarity score (0-1)

    Returns:
        str: Formatted similarity with color
    """
    if similarity is None:
        return "—"

    percentage = similarity * 100
    if similarity >= 0.9:
        color = "green"
    elif similarity >= 0.7:
        color = "yellow"
    else:
        color = "red"

    return f"[{color}]{percentage:.1f}%[/{color}]"


def truncate_text(text: str, max_length: int = 80) -> str:
    """Truncate text to maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        str: Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."
