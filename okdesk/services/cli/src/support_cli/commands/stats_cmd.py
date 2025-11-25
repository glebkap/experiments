"""Statistics commands."""

from datetime import date
from typing import Literal

import typer
from rich.console import Console

from support_cli.client.api_client import APIClient
from support_cli.client.exceptions import APIError
from support_cli.client.models import (
    ClusterStats,
    ProcessingStats,
    SourceStats,
    TimelineStats,
)
from support_cli.output.formatters import format_percentage
from support_cli.output.tables import create_source_stats_table, create_stats_table

app = typer.Typer(help="Statistics and analytics")
console = Console()


@app.command()
def processing(ctx: typer.Context) -> None:
    """Show processing statistics.

    Displays information about processed and pending issues, success rate,
    and average processing time.

    Example:
        support-cli stats processing
    """
    client: APIClient = ctx.obj["client"]

    try:
        response = client.get_stats_processing()

        if isinstance(response.data, ProcessingStats):
            stats = response.data

            stats_data = {
                "Total Issues": stats.total_issues,
                "Processed": stats.processed,
                "Pending": stats.pending,
                "Success Rate": format_percentage(stats.success_rate),
            }

            if stats.average_processing_time:
                stats_data["Avg Processing Time"] = f"{stats.average_processing_time:.2f}s"

            table = create_stats_table("Processing Statistics", stats_data)
            console.print(table)

            # Visual indicator
            if stats.pending > 0:
                console.print(
                    f"\n[yellow]⚠ {stats.pending} issues pending processing[/yellow]"
                )
            else:
                console.print("\n[success]✓ All issues processed[/success]")

        else:
            console.print("[warning]Unexpected response format[/warning]")

    except APIError as e:
        console.print(f"[error]Error:[/error] {e.message}", style="bold")
        raise typer.Exit(1)


@app.command()
def clusters(ctx: typer.Context) -> None:
    """Show cluster statistics.

    Displays information about clusters including total count, sizes,
    and distribution.

    Example:
        support-cli stats clusters
    """
    client: APIClient = ctx.obj["client"]

    try:
        response = client.get_stats_clusters()

        if isinstance(response.data, ClusterStats):
            stats = response.data

            stats_data = {
                "Total Clusters": stats.total_clusters,
                "Issues Clustered": stats.total_issues_clustered,
                "Noise Points": stats.noise_points,
                "Largest Cluster": stats.largest_cluster_size,
                "Average Cluster Size": f"{stats.average_cluster_size:.1f}",
            }

            table = create_stats_table("Cluster Statistics", stats_data)
            console.print(table)

            if stats.top_clusters:
                console.print("\n[bold]Top Clusters by Size:[/bold]\n")
                for i, cluster in enumerate(stats.top_clusters[:10], 1):
                    name = cluster.name or f"Cluster {cluster.cluster_label}"
                    console.print(f"{i:2d}. [cyan]{name}[/cyan]: {cluster.size} issues")

        else:
            console.print("[warning]Unexpected response format[/warning]")

    except APIError as e:
        console.print(f"[error]Error:[/error] {e.message}", style="bold")
        raise typer.Exit(1)


@app.command()
def sources(ctx: typer.Context) -> None:
    """Show statistics by data sources.

    Displays information about each data source including total issues,
    processed count, and progress.

    Example:
        support-cli stats sources
    """
    client: APIClient = ctx.obj["client"]

    try:
        response = client.get_stats_sources()

        if isinstance(response.data, list):
            sources = [SourceStats.model_validate(s) for s in response.data]

            if not sources:
                console.print("[warning]No sources found[/warning]")
                return

            table = create_source_stats_table(sources)
            console.print(table)

            # Summary
            total_issues = sum(s.total_issues for s in sources)
            total_processed = sum(s.processed for s in sources)
            console.print(
                f"\n[bold]Total:[/bold] {total_processed:,} / {total_issues:,} "
                f"({format_percentage(total_processed / total_issues if total_issues > 0 else 0)})"
            )

        else:
            console.print("[warning]Unexpected response format[/warning]")

    except APIError as e:
        console.print(f"[error]Error:[/error] {e.message}", style="bold")
        raise typer.Exit(1)


@app.command()
def timeline(
    ctx: typer.Context,
    from_date: str = typer.Option(None, "--from", help="Start date (YYYY-MM-DD)"),
    to_date: str = typer.Option(None, "--to", help="End date (YYYY-MM-DD)"),
    granularity: Literal["day", "week", "month"] = typer.Option(
        "day",
        "--granularity",
        "-g",
        help="Time granularity",
    ),
) -> None:
    """Show timeline statistics.

    Displays issue counts over time with specified granularity.

    Example:
        support-cli stats timeline --from 2025-01-01 --to 2025-11-25 --granularity week
    """
    client: APIClient = ctx.obj["client"]

    try:
        # Validate dates
        if from_date:
            date.fromisoformat(from_date)
        if to_date:
            date.fromisoformat(to_date)

        response = client.get_stats_timeline(
            from_date=from_date,
            to_date=to_date,
            granularity=granularity,
        )

        if isinstance(response.data, TimelineStats):
            stats = response.data

            console.print(f"\n[bold]Timeline Statistics[/bold]", style="header")
            console.print(f"Period: {stats.from_date.date()} to {stats.to_date.date()}")
            console.print(f"Granularity: {stats.granularity}\n")

            if not stats.data_points:
                console.print("[warning]No data points found[/warning]")
                return

            # Show data points
            for point in stats.data_points:
                # Create simple bar chart
                bar_length = min(50, point.count // 10)  # Scale for display
                bar = "█" * bar_length
                console.print(f"{point.period:12s} {bar} {point.count:,}")

            # Summary
            total = sum(p.count for p in stats.data_points)
            avg = total / len(stats.data_points)
            console.print(
                f"\n[bold]Total:[/bold] {total:,} issues, "
                f"[bold]Average:[/bold] {avg:.1f} per {granularity}"
            )

        else:
            console.print("[warning]Unexpected response format[/warning]")

    except ValueError as e:
        console.print(f"[error]Invalid date format:[/error] {e}", style="bold")
        console.print("Use YYYY-MM-DD format")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[error]Error:[/error] {e.message}", style="bold")
        raise typer.Exit(1)
