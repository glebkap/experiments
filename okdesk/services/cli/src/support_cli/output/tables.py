"""Rich table generators."""

from typing import Any

from rich.table import Table

from support_cli.client.models import (
    ClusterInfo,
    ImportHistory,
    IssueListItem,
    SearchResult,
    SourceStats,
)

from .formatters import (
    format_datetime,
    format_percentage,
    format_similarity,
    format_status,
    truncate_text,
)


def create_imports_table(imports: list[ImportHistory]) -> Table:
    """Create table for import history.

    Args:
        imports: List of import history items

    Returns:
        Table: Rich table
    """
    table = Table(title="Import History", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="dim", width=36)
    table.add_column("Filename", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Started", justify="center")
    table.add_column("Completed", justify="center")
    table.add_column("Issues", justify="right")
    table.add_column("Messages", justify="right")

    for imp in imports:
        status = format_status(imp.status)
        issues = str(imp.stats.total_issues) if imp.stats else "—"
        messages = str(imp.stats.total_messages) if imp.stats else "—"

        table.add_row(
            imp.id,
            imp.filename or "—",
            status,
            format_datetime(imp.started_at),
            format_datetime(imp.completed_at),
            issues,
            messages,
        )

    return table


def create_issues_table(issues: list[IssueListItem], show_similarity: bool = False) -> Table:
    """Create table for issues list.

    Args:
        issues: List of issues
        show_similarity: Whether to show similarity column

    Returns:
        Table: Rich table
    """
    table = Table(title="Issues", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="dim", width=36)
    table.add_column("Title", style="cyan", max_width=50)
    table.add_column("Status", justify="center", width=12)
    table.add_column("Priority", justify="center", width=8)
    table.add_column("Created", justify="center", width=19)

    if show_similarity:
        table.add_column("Similarity", justify="center", width=10)

    for issue in issues:
        status = format_status(issue.status) if issue.status else "—"
        priority_str = str(issue.priority) if issue.priority else "—"
        title = truncate_text(issue.title, 47) if issue.title else "—"

        row = [
            issue.id,
            title,
            status,
            priority_str,
            format_datetime(issue.created_at),
        ]

        if show_similarity:
            row.append(format_similarity(issue.similarity))

        table.add_row(*row)

    return table


def create_search_results_table(results: list[SearchResult]) -> Table:
    """Create table for search results.

    Args:
        results: List of search results

    Returns:
        Table: Rich table
    """
    table = Table(title="Search Results", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="dim", width=36)
    table.add_column("Similarity", justify="center", width=10)
    table.add_column("Title", style="cyan", max_width=30)
    table.add_column("Snippet", max_width=50)

    for result in results:
        table.add_row(
            result.issue_id,
            format_similarity(result.similarity),
            truncate_text(result.title, 27) if result.title else "—",
            truncate_text(result.text_snippet, 47),
        )

    return table


def create_clusters_table(clusters: list[ClusterInfo]) -> Table:
    """Create table for clusters.

    Args:
        clusters: List of clusters

    Returns:
        Table: Rich table
    """
    table = Table(title="Clusters", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="dim", width=36)
    table.add_column("Label", justify="center", width=8)
    table.add_column("Name", style="cyan", max_width=30)
    table.add_column("Size", justify="right", width=10)
    table.add_column("Created", justify="center", width=19)

    for cluster in clusters:
        table.add_row(
            cluster.id,
            str(cluster.cluster_label),
            truncate_text(cluster.name, 27) if cluster.name else "—",
            str(cluster.size),
            format_datetime(cluster.created_at),
        )

    return table


def create_stats_table(title: str, data: dict[str, Any]) -> Table:
    """Create generic statistics table.

    Args:
        title: Table title
        data: Statistics data as key-value pairs

    Returns:
        Table: Rich table
    """
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="bold")

    for key, value in data.items():
        # Format value based on type
        if isinstance(value, float) and 0 <= value <= 1:
            formatted_value = format_percentage(value)
        elif isinstance(value, int):
            formatted_value = f"{value:,}"
        else:
            formatted_value = str(value)

        table.add_row(key, formatted_value)

    return table


def create_source_stats_table(sources: list[SourceStats]) -> Table:
    """Create table for source statistics.

    Args:
        sources: List of source statistics

    Returns:
        Table: Rich table
    """
    table = Table(title="Source Statistics", show_header=True, header_style="bold magenta")
    table.add_column("Source Name", style="cyan")
    table.add_column("Type", justify="center", width=12)
    table.add_column("Total", justify="right", width=10)
    table.add_column("Processed", justify="right", width=10)
    table.add_column("Pending", justify="right", width=10)
    table.add_column("Progress", justify="center", width=10)

    for source in sources:
        progress = (
            source.processed / source.total_issues if source.total_issues > 0 else 0
        )

        table.add_row(
            source.source_name,
            source.source_type,
            str(source.total_issues),
            str(source.processed),
            str(source.pending),
            format_percentage(progress),
        )

    return table
