"""Import commands for CLI."""

import time
from pathlib import Path

import typer
from rich.console import Console

from support_cli.client.api_client import APIClient
from support_cli.client.exceptions import APIError
from support_cli.output.formatters import format_datetime, format_status
from support_cli.output.progress import create_spinner
from support_cli.output.tables import create_imports_table

app = typer.Typer(help="Import data from various sources")
console = Console()


@app.command()
def okdesk(
    ctx: typer.Context,
    file: Path = typer.Argument(..., help="Path to OKDesk JSONL file", exists=True),
    wait: bool = typer.Option(False, "--wait", "-w", help="Wait for import to complete"),
) -> None:
    """Import data from OKDesk JSONL file.

    Example:
        support-cli import okdesk /data/okdesk/out.jsonl --wait
    """
    client: APIClient = ctx.obj["client"]

    try:
        console.print(f"[cyan]Uploading file:[/cyan] {file}")

        response = client.import_okdesk(file)

        console.print(
            f"[success]Import started:[/success] {response.import_id}",
            style="bold",
        )
        console.print(f"Status: {format_status(response.status)}")

        if wait:
            _wait_for_import(client, response.import_id)

    except APIError as e:
        console.print(f"[error]Error:[/error] {e.message}", style="bold")
        raise typer.Exit(1)


@app.command()
def telegram(
    ctx: typer.Context,
    file: Path = typer.Argument(..., help="Path to Telegram JSON file", exists=True),
    wait: bool = typer.Option(False, "--wait", "-w", help="Wait for import to complete"),
) -> None:
    """Import data from Telegram JSON file.

    Example:
        support-cli import telegram /data/telegram/result.json
    """
    client: APIClient = ctx.obj["client"]

    try:
        console.print(f"[cyan]Uploading file:[/cyan] {file}")

        response = client.import_telegram(file)

        console.print(
            f"[success]Import started:[/success] {response.import_id}",
            style="bold",
        )
        console.print(f"Status: {format_status(response.status)}")

        if wait:
            _wait_for_import(client, response.import_id)

    except APIError as e:
        console.print(f"[error]Error:[/error] {e.message}", style="bold")
        raise typer.Exit(1)


@app.command()
def status(
    ctx: typer.Context,
    import_id: str = typer.Argument(..., help="Import ID"),
) -> None:
    """Show import status by ID.

    Example:
        support-cli import status abc-123-def
    """
    client: APIClient = ctx.obj["client"]

    try:
        import_status = client.get_import_status(import_id)

        console.print(f"\n[bold]Import Status[/bold]", style="header")
        console.print(f"ID: [dim]{import_status.id}[/dim]")
        console.print(f"Status: {format_status(import_status.status)}")
        console.print(f"Started: {format_datetime(import_status.started_at)}")
        console.print(f"Completed: {format_datetime(import_status.completed_at)}")

        if import_status.stats:
            console.print("\n[bold]Statistics:[/bold]")
            console.print(f"  Total Issues: {import_status.stats.total_issues}")
            console.print(f"  New Issues: {import_status.stats.new_issues}")
            console.print(f"  Updated Issues: {import_status.stats.updated_issues}")
            console.print(f"  Total Messages: {import_status.stats.total_messages}")
            console.print(f"  New Messages: {import_status.stats.new_messages}")
            console.print(
                f"  Skipped Duplicates: {import_status.stats.skipped_duplicates}"
            )

        if import_status.error_message:
            console.print(f"\n[error]Error:[/error] {import_status.error_message}")

    except APIError as e:
        console.print(f"[error]Error:[/error] {e.message}", style="bold")
        raise typer.Exit(1)


@app.command()
def history(
    ctx: typer.Context,
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum number of records"),
) -> None:
    """Show import history.

    Example:
        support-cli import history --limit 10
    """
    client: APIClient = ctx.obj["client"]

    try:
        imports = client.list_imports(limit=limit)

        if not imports:
            console.print("[warning]No imports found[/warning]")
            return

        table = create_imports_table(imports)
        console.print(table)

    except APIError as e:
        console.print(f"[error]Error:[/error] {e.message}", style="bold")
        raise typer.Exit(1)


def _wait_for_import(client: APIClient, import_id: str, check_interval: float = 2.0) -> None:
    """Wait for import to complete with spinner.

    Args:
        client: API client
        import_id: Import ID
        check_interval: Seconds between status checks
    """
    with create_spinner("Waiting for import to complete...") as progress:
        task = progress.add_task("Import in progress...", total=None)

        while True:
            import_status = client.get_import_status(import_id)

            if import_status.status == "completed":
                progress.update(task, description="[success]Import completed![/success]")
                console.print("\n[success]Import completed successfully![/success]", style="bold")

                if import_status.stats:
                    console.print(f"Total Issues: {import_status.stats.total_issues}")
                    console.print(f"New Issues: {import_status.stats.new_issues}")
                    console.print(f"Total Messages: {import_status.stats.total_messages}")
                break

            elif import_status.status == "failed":
                progress.update(task, description="[error]Import failed[/error]")
                console.print("\n[error]Import failed[/error]", style="bold")
                if import_status.error_message:
                    console.print(f"Error: {import_status.error_message}")
                raise typer.Exit(1)

            time.sleep(check_interval)
