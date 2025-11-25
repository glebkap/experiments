"""Export commands."""

from pathlib import Path
from typing import Any, Literal

import typer
from rich.console import Console

from support_cli.client.api_client import APIClient
from support_cli.client.exceptions import APIError
from support_cli.output.formatters import format_file_size

app = typer.Typer(help="Export data operations")
console = Console()


@app.command()
def data(
    ctx: typer.Context,
    format: Literal["csv", "json"] = typer.Option(
        "csv",
        "--format",
        "-f",
        help="Export format",
    ),
    output: Path = typer.Option(None, "--output", "-o", help="Output file path"),
    status: str = typer.Option(None, "--status", help="Filter by status"),
    cluster_id: str = typer.Option(None, "--cluster", help="Filter by cluster ID"),
    from_date: str = typer.Option(None, "--from", help="Filter from date (YYYY-MM-DD)"),
    to_date: str = typer.Option(None, "--to", help="Filter to date (YYYY-MM-DD)"),
) -> None:
    """Export data with filters.

    Exports issues and their data based on specified filters and format.

    Example:
        support-cli export data --format csv --output export.csv --status opened
        support-cli export data --format json --cluster <id> --output cluster.json
    """
    client: APIClient = ctx.obj["client"]

    try:
        # Build filters dict
        filters: dict[str, Any] = {}
        if status:
            filters["status"] = status
        if cluster_id:
            filters["cluster_id"] = cluster_id
        if from_date:
            filters["from_date"] = from_date
        if to_date:
            filters["to_date"] = to_date

        # Show export info
        console.print("[cyan]Exporting data...[/cyan]")
        console.print(f"Format: {format}")

        if filters:
            console.print("Filters:")
            for key, value in filters.items():
                console.print(f"  - {key}: {value}")

        console.print()

        # Call API
        result = client.export_data(format=format, filters=filters)

        console.print("[success]Export completed![/success]\n", style="bold")
        console.print(f"Filename: [cyan]{result.filename}[/cyan]")
        console.print(f"Format: {result.format}")
        console.print(f"Total records: [bold]{result.total_records}[/bold]")

        if result.file_size_bytes:
            console.print(f"File size: {format_file_size(result.file_size_bytes)}")

        # Handle output
        if output:
            # If API provides download_url, we'd download it here
            # For now, just show the information
            console.print(f"\n[dim]Output path: {output}[/dim]")

            if result.download_url:
                console.print(f"[dim]Download URL: {result.download_url}[/dim]")
                console.print(
                    "\n[yellow]Note: Actual file download not implemented yet[/yellow]"
                )
        elif result.download_url:
            console.print(f"\n[cyan]Download URL:[/cyan] {result.download_url}")

    except APIError as e:
        console.print(f"[error]Error:[/error] {e.message}", style="bold")
        raise typer.Exit(1)
