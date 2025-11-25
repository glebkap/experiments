"""Progress bars and spinners for long-running operations."""

import time
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def create_progress() -> Progress:
    """Create a configured Progress instance.

    Returns:
        Progress: Configured progress bar
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=Console(),
    )


def create_spinner(description: str = "Processing...") -> Progress:
    """Create a spinner for indeterminate progress.

    Args:
        description: Description text

    Returns:
        Progress: Configured spinner
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=Console(),
    )


def wait_for_completion(
    check_status_fn: Any,
    check_interval: float = 2.0,
    timeout: float = 300.0,
    description: str = "Waiting for completion...",
) -> Any:
    """Wait for operation to complete with progress indicator.

    Args:
        check_status_fn: Function that returns status dict with 'completed' boolean
        check_interval: Seconds between status checks
        timeout: Maximum time to wait in seconds
        description: Progress description

    Returns:
        Any: Final status result

    Raises:
        TimeoutError: If operation doesn't complete within timeout
    """
    with create_spinner(description) as progress:
        task = progress.add_task(description, total=None)

        start_time = time.time()
        while True:
            status = check_status_fn()

            if status.get("completed", False):
                progress.update(task, completed=True)
                return status

            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Operation timed out after {timeout} seconds")

            time.sleep(check_interval)
