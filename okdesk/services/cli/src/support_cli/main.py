"""Main CLI application entry point."""

import sys
from typing import Optional

import typer
from rich.console import Console

from support_cli.client.api_client import APIClient
from support_cli.client.exceptions import APIError, ConnectionError
from support_cli.commands import (
    clustering_cmd,
    export_cmd,
    import_cmd,
    pipeline_cmd,
    search_cmd,
    stats_cmd,
)
from support_cli.config import get_settings
from support_cli.output.styles import CLI_THEME

# Create main app
app = typer.Typer(
    name="support-cli",
    help="CLI для системы анализа обращений службы поддержки",
    add_completion=True,
    rich_markup_mode="rich",
)

# Register command groups
app.add_typer(import_cmd.app, name="import", help="Import data from various sources")
app.add_typer(pipeline_cmd.app, name="pipeline", help="Pipeline processing operations")
app.add_typer(clustering_cmd.app, name="cluster", help="Clustering operations")
app.add_typer(search_cmd.app, name="search", help="Search and view operations")
app.add_typer(stats_cmd.app, name="stats", help="Statistics and analytics")
app.add_typer(export_cmd.app, name="export", help="Export data operations")

console = Console(theme=CLI_THEME)


@app.callback()
def main(
    ctx: typer.Context,
    api_url: Optional[str] = typer.Option(
        None,
        "--api-url",
        envvar="API_GATEWAY_URL",
        help="API Gateway URL",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress non-error output",
    ),
) -> None:
    """Support CLI - командный интерфейс для системы анализа обращений.

    Используйте подкоманды для различных операций:

    \b
    • import   - Импорт данных из OKDesk и Telegram
    • pipeline - Запуск pipeline обработки issues
    • cluster  - Кластеризация схожих issues
    • search   - Поиск и просмотр issues
    • stats    - Статистика и аналитика
    • export   - Экспорт данных

    Примеры:

        support-cli import okdesk /data/okdesk/out.jsonl

        support-cli pipeline process --batch-size 100

        support-cli cluster run --method hdbscan

        support-cli search similar "проблема с оплатой"

        support-cli stats processing
    """
    # Load settings
    settings = get_settings()

    # Override API URL if provided
    if api_url:
        settings.api_gateway_url = api_url  # type: ignore

    # Set verbosity
    if quiet:
        console.quiet = True

    # Initialize API client
    try:
        client = APIClient(settings)

        # Store client in context for subcommands
        ctx.obj = {"client": client, "settings": settings, "verbose": verbose}

        # Check API health if not running help command
        if ctx.invoked_subcommand and ctx.invoked_subcommand not in ["--help", "-h"]:
            if not client.health_check():
                console.print(
                    f"[warning]⚠ Warning: Cannot reach API at {settings.api_gateway_url}[/warning]"
                )
                console.print(
                    "[dim]Some commands may fail. Check your API_GATEWAY_URL setting.[/dim]\n"
                )

    except Exception as e:
        console.print(f"[error]Failed to initialize API client:[/error] {e}", style="bold")
        raise typer.Exit(1)


@app.command()
def health(ctx: typer.Context) -> None:
    """Check API health status.

    Example:
        support-cli health
    """
    client: APIClient = ctx.obj["client"]
    settings = ctx.obj["settings"]

    console.print(f"[cyan]Checking API health at {settings.api_gateway_url}...[/cyan]\n")

    try:
        if client.health_check():
            console.print("[success]✓ API is healthy[/success]", style="bold")
        else:
            console.print("[error]✗ API is not responding[/error]", style="bold")
            raise typer.Exit(1)

    except ConnectionError as e:
        console.print(f"[error]✗ Connection failed:[/error] {e.message}", style="bold")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[error]✗ API error:[/error] {e.message}", style="bold")
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Show CLI version information."""
    console.print("[bold]support-cli[/bold] version [cyan]0.1.0[/cyan]")
    console.print("Система анализа обращений службы поддержки")


def cli_entry_point() -> None:
    """Entry point for console script."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[warning]Interrupted by user[/warning]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[error]Unexpected error:[/error] {e}", style="bold")
        sys.exit(1)


if __name__ == "__main__":
    cli_entry_point()
