"""Date utilities."""

from datetime import datetime


def parse_date(date_str: str) -> datetime:
    """Parse date string in ISO format.

    Args:
        date_str: Date string in YYYY-MM-DD format

    Returns:
        datetime: Parsed datetime object

    Raises:
        ValueError: If date format is invalid
    """
    return datetime.fromisoformat(date_str)


def format_date_iso(dt: datetime) -> str:
    """Format datetime to ISO date string.

    Args:
        dt: Datetime object

    Returns:
        str: ISO formatted date string (YYYY-MM-DD)
    """
    return dt.date().isoformat()
