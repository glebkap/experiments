"""Pipeline processing commands."""

import typer
from rich.console import Console

from support_cli.client.api_client import APIClient
from support_cli.client.exceptions import APIError
from support_cli.output.formatters import format_datetime, format_duration, format_percentage
from support_cli.output.tables import create_stats_table

app = typer.Typer(help="Pipeline processing operations")
console = Console()


@app.command()
def process(
    ctx: typer.Context,
    batch_size: int = typer.Option(
        100,
        "--batch-size",
        "-b",
        help="Batch size for processing",
        min=1,
        max=1000,
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        "-d",
        help="Device to use (cpu/cuda)",
    ),
) -> None:
    """Start pipeline processing of unprocessed issues.

    This will process issues through all pipeline stages:
    1. Preprocessing (HTML cleanup, lemmatization)
    2. Embedding generation (SentenceTransformers)
    3. Vector DB storage (ChromaDB)
    4. Completion

    Example:
        support-cli pipeline process --batch-size 100 --device cpu
    """
    client: APIClient = ctx.obj["client"]

    try:
        console.print(f"[cyan]Starting pipeline processing...[/cyan]")
        console.print(f"Batch size: {batch_size}")
        console.print(f"Device: {device}\n")

        response = client.process_pipeline(batch_size=batch_size, device=device)

        console.print("[success]Pipeline processing completed![/success]\n", style="bold")
        console.print(f"Processed: [bold]{response.processed_count}[/bold] issues")
        console.print(f"Duration: {format_duration(response.duration_seconds)}\n")

        if response.stats:
            stats_data = {
                "Stage 1 (Preprocessed)": response.stats.stage_1_preprocessed,
                "Stage 2 (Embeddings)": response.stats.stage_2_embeddings,
                "Stage 3 (Vector DB)": response.stats.stage_3_vector_db,
                "Stage 4 (Completed)": response.stats.stage_4_completed,
                "Failed": response.stats.failed,
            }

            if response.stats.total_duration_seconds:
                stats_data["Total Duration"] = format_duration(
                    response.stats.total_duration_seconds
                )

            table = create_stats_table("Pipeline Statistics", stats_data)
            console.print(table)

    except APIError as e:
        console.print(f"[error]Error:[/error] {e.message}", style="bold")
        raise typer.Exit(1)


@app.command()
def status(ctx: typer.Context) -> None:
    """Show pipeline processing status.

    Displays statistics about processed and pending issues.

    Example:
        support-cli pipeline status
    """
    client: APIClient = ctx.obj["client"]

    try:
        status_data = client.get_pipeline_status()

        console.print("\n[bold]Pipeline Status[/bold]", style="header")

        stats_data = {
            "Total Issues": status_data.total_issues,
            "Processed": status_data.processed_issues,
            "Pending": status_data.pending_issues,
        }

        if status_data.success_rate is not None:
            stats_data["Success Rate"] = format_percentage(status_data.success_rate)

        if status_data.last_processed_at:
            stats_data["Last Processed"] = format_datetime(status_data.last_processed_at)

        table = create_stats_table("Processing Statistics", stats_data)
        console.print(table)

        # Show pending count prominently
        if status_data.pending_issues > 0:
            console.print(
                f"\n[yellow]⚠ {status_data.pending_issues} issues pending processing[/yellow]"
            )
        else:
            console.print("\n[success]✓ All issues processed[/success]")

    except APIError as e:
        console.print(f"[error]Error:[/error] {e.message}", style="bold")
        raise typer.Exit(1)
