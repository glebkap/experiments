"""Search and view commands."""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from support_cli.client.api_client import APIClient
from support_cli.client.exceptions import APIError
from support_cli.output.formatters import format_datetime, format_json_pretty, format_status
from support_cli.output.tables import create_issues_table, create_search_results_table

app = typer.Typer(help="Search and view operations")
console = Console()


@app.command()
def similar(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="Search query text"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results", min=1, max=100),
    min_similarity: float = typer.Option(
        0.7,
        "--min-similarity",
        "-s",
        help="Minimum similarity threshold",
        min=0.0,
        max=1.0,
    ),
) -> None:
    """Semantic search for similar issues using embeddings.

    Finds issues semantically similar to the query text using vector similarity.

    Example:
        support-cli search similar "проблема с оплатой" --top-k 10
    """
    client: APIClient = ctx.obj["client"]

    try:
        console.print(f"[cyan]Searching for:[/cyan] {query}")
        console.print(f"Top K: {top_k}, Min similarity: {min_similarity}\n")

        results = client.search_similar(
            query=query,
            top_k=top_k,
            min_similarity=min_similarity,
        )

        if not results:
            console.print("[warning]No similar issues found[/warning]")
            console.print("Try lowering the --min-similarity threshold")
            return

        console.print(f"[success]Found {len(results)} similar issues:[/success]\n")
        table = create_search_results_table(results)
        console.print(table)

    except APIError as e:
        console.print(f"[error]Error:[/error] {e.message}", style="bold")
        raise typer.Exit(1)


@app.command()
def fulltext(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="Search query text"),
    limit: int = typer.Option(50, "--limit", "-l", help="Maximum results", min=1),
    offset: int = typer.Option(0, "--offset", "-o", help="Offset for pagination", min=0),
) -> None:
    """Full-text search in issues using PostgreSQL.

    Searches through issue titles and descriptions using PostgreSQL full-text search.

    Example:
        support-cli search fulltext "платеж не прошел" --limit 20
    """
    client: APIClient = ctx.obj["client"]

    try:
        console.print(f"[cyan]Searching for:[/cyan] {query}\n")

        results = client.search_fulltext(query=query, limit=limit, offset=offset)

        if not results:
            console.print("[warning]No issues found[/warning]")
            return

        console.print(f"[success]Found {len(results)} issues:[/success]\n")
        table = create_issues_table(results)
        console.print(table)

        if len(results) == limit:
            console.print(
                f"\n[dim]Showing {limit} results. "
                f"Use --offset {offset + limit} to see more.[/dim]"
            )

    except APIError as e:
        console.print(f"[error]Error:[/error] {e.message}", style="bold")
        raise typer.Exit(1)


@app.command(name="list")
def list_issues(
    ctx: typer.Context,
    status: str = typer.Option(None, "--status", "-s", help="Filter by status"),
    source_id: str = typer.Option(None, "--source", help="Filter by source ID"),
    limit: int = typer.Option(50, "--limit", "-l", help="Maximum results", min=1),
    offset: int = typer.Option(0, "--offset", "-o", help="Offset for pagination", min=0),
) -> None:
    """List issues with optional filters.

    Example:
        support-cli search list --status opened --limit 50
        support-cli search list --source <source-id>
    """
    client: APIClient = ctx.obj["client"]

    try:
        results = client.list_issues(
            status=status,
            source_id=source_id,
            limit=limit,
            offset=offset,
        )

        if not results:
            console.print("[warning]No issues found[/warning]")
            return

        # Show filter info
        filters = []
        if status:
            filters.append(f"status={status}")
        if source_id:
            filters.append(f"source={source_id[:8]}...")

        if filters:
            console.print(f"[dim]Filters: {', '.join(filters)}[/dim]\n")

        console.print(f"[success]Found {len(results)} issues:[/success]\n")
        table = create_issues_table(results)
        console.print(table)

        if len(results) == limit:
            console.print(
                f"\n[dim]Showing {limit} results. "
                f"Use --offset {offset + limit} to see more.[/dim]"
            )

    except APIError as e:
        console.print(f"[error]Error:[/error] {e.message}", style="bold")
        raise typer.Exit(1)


@app.command()
def show(
    ctx: typer.Context,
    issue_id: str = typer.Argument(..., help="Issue ID"),
    show_messages: bool = typer.Option(
        True,
        "--messages/--no-messages",
        help="Show messages",
    ),
    format: str = typer.Option(
        "pretty",
        "--format",
        "-f",
        help="Output format (pretty/json)",
    ),
) -> None:
    """Show detailed issue information.

    Example:
        support-cli search show <issue-id>
        support-cli search show <issue-id> --format json
    """
    client: APIClient = ctx.obj["client"]

    try:
        issue = client.get_issue(issue_id)

        if format == "json":
            # JSON output
            json_str = format_json_pretty(issue.model_dump())
            syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
            console.print(syntax)
            return

        # Pretty output
        console.print(f"\n[bold]Issue Details[/bold]", style="header")
        console.print(f"ID: [dim]{issue.id}[/dim]")
        console.print(f"External ID: {issue.external_id}")
        console.print(f"Title: [cyan]{issue.title or '—'}[/cyan]")

        if issue.status:
            console.print(f"Status: {format_status(issue.status)}")

        console.print(f"Priority: {issue.priority or '—'}")
        console.print(f"Created: {format_datetime(issue.created_at)}")
        console.print(f"Updated: {format_datetime(issue.updated_at)}")
        console.print(f"Completed: {format_datetime(issue.completed_at)}")

        if issue.cluster_id:
            console.print(f"Cluster: [dim]{issue.cluster_id}[/dim]")

        if issue.description:
            console.print(f"\n[bold]Description:[/bold]")
            # Truncate long descriptions
            desc = issue.description[:500]
            if len(issue.description) > 500:
                desc += "..."
            console.print(Panel(desc, border_style="dim"))

        if show_messages and issue.messages:
            console.print(f"\n[bold]Messages ({len(issue.messages)}):[/bold]\n")

            for i, msg in enumerate(issue.messages, 1):
                author = f"{msg.author_name} ({msg.author_type})" if msg.author_name else "—"
                console.print(f"[bold]{i}. {author}[/bold] - {format_datetime(msg.published_at)}")

                # Truncate long messages
                content = msg.content[:200]
                if len(msg.content) > 200:
                    content += "..."

                console.print(f"   {content}\n")

    except APIError as e:
        console.print(f"[error]Error:[/error] {e.message}", style="bold")
        raise typer.Exit(1)
