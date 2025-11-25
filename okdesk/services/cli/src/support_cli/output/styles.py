"""Rich styling and color schemes."""

from rich.style import Style
from rich.theme import Theme

# Status colors
STATUS_COLORS = {
    "success": "green",
    "error": "red",
    "warning": "yellow",
    "info": "blue",
    "in_progress": "cyan",
    "completed": "green",
    "failed": "red",
    "opened": "yellow",
    "wait": "cyan",
    "closed": "dim",
}

# Custom theme
CLI_THEME = Theme(
    {
        "success": "bold green",
        "error": "bold red",
        "warning": "bold yellow",
        "info": "bold blue",
        "dim": "dim",
        "highlight": "bold cyan",
        "header": "bold magenta",
        "value": "cyan",
    }
)


def get_status_style(status: str) -> Style:
    """Get Rich style for status.

    Args:
        status: Status string

    Returns:
        Style: Rich style object
    """
    color = STATUS_COLORS.get(status.lower(), "white")
    return Style(color=color, bold=True)
