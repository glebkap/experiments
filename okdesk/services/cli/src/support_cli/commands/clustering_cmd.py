"""Clustering commands."""

from typing import Literal

import typer
from rich.console import Console

from support_cli.client.api_client import APIClient
from support_cli.client.exceptions import APIError
from support_cli.output.tables import create_clusters_table, create_issues_table

app = typer.Typer(help="Clustering operations", name="cluster")
console = Console()


@app.command()
def run(
    ctx: typer.Context,
    method: Literal["hdbscan", "kmeans"] = typer.Option(
        "hdbscan",
        "--method",
        "-m",
        help="Clustering algorithm",
    ),
    min_cluster_size: int = typer.Option(
        5,
        "--min-size",
        "-s",
        help="Minimum cluster size",
        min=2,
    ),
    use_llm: bool = typer.Option(
        False,
        "--use-llm",
        help="Use LLM to generate cluster names and descriptions",
    ),
) -> None:
    """Run clustering algorithm on processed issues.

    HDBSCAN automatically determines the number of clusters based on density.
    K-means requires specifying the number of clusters.

    Example:
        support-cli cluster run --method hdbscan --min-size 5
        support-cli cluster run --method hdbscan --use-llm
    """
    client: APIClient = ctx.obj["client"]

    try:
        console.print(f"[cyan]Starting clustering...[/cyan]")
        console.print(f"Method: {method}")
        console.print(f"Min cluster size: {min_cluster_size}")
        console.print(f"Generate names: {use_llm}\n")

        result = client.run_clustering(
            method=method,
            min_cluster_size=min_cluster_size,
            use_llm=use_llm,
        )

        console.print("[success]Clustering completed![/success]\n", style="bold")
        console.print(f"Total clusters: [bold]{result.total_clusters}[/bold]")
        console.print(f"Issues clustered: [bold]{result.total_issues_clustered}[/bold]")
        console.print(f"Noise points: [bold]{result.noise_points}[/bold]\n")

        if result.clusters:
            table = create_clusters_table(result.clusters[:10])  # Show top 10
            console.print(table)

            if len(result.clusters) > 10:
                console.print(
                    f"\n[dim]Showing 10 of {len(result.clusters)} clusters. "
                    f"Use 'cluster list' to see all.[/dim]"
                )

    except APIError as e:
        console.print(f"[error]Error:[/error] {e.message}", style="bold")
        raise typer.Exit(1)


@app.command()
def list(ctx: typer.Context) -> None:
    """List all clusters.

    Example:
        support-cli cluster list
    """
    client: APIClient = ctx.obj["client"]

    try:
        result = client.get_clustering_info()

        if result.total_clusters == 0:
            console.print("[warning]No clusters found. Run clustering first.[/warning]")
            return

        console.print(f"\n[bold]Total clusters:[/bold] {result.total_clusters}")
        console.print(f"[bold]Issues clustered:[/bold] {result.total_issues_clustered}")
        console.print(f"[bold]Noise points:[/bold] {result.noise_points}\n")

        if result.clusters:
            table = create_clusters_table(result.clusters)
            console.print(table)

    except APIError as e:
        console.print(f"[error]Error:[/error] {e.message}", style="bold")
        raise typer.Exit(1)


@app.command()
def show(
    ctx: typer.Context,
    cluster_id: str = typer.Argument(..., help="Cluster ID"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of issues to show"),
) -> None:
    """Show issues in a specific cluster.

    Example:
        support-cli cluster show <cluster-id> --limit 20
    """
    client: APIClient = ctx.obj["client"]

    try:
        result = client.get_cluster_issues(cluster_id=cluster_id, limit=limit)

        cluster = result.cluster
        console.print(f"\n[bold]Cluster Details[/bold]", style="header")
        console.print(f"ID: [dim]{cluster.id}[/dim]")
        console.print(f"Label: {cluster.cluster_label}")
        console.print(f"Name: {cluster.name or '—'}")
        console.print(f"Size: [bold]{cluster.size}[/bold] issues")

        if cluster.description:
            console.print(f"Description: {cluster.description}")

        console.print(f"\n[bold]Top Issues in Cluster:[/bold]\n")

        if result.issues:
            table = create_issues_table(result.issues)
            console.print(table)

            if cluster.size > limit:
                console.print(
                    f"\n[dim]Showing {limit} of {cluster.size} issues in cluster.[/dim]"
                )
        else:
            console.print("[warning]No issues found in cluster[/warning]")

    except APIError as e:
        console.print(f"[error]Error:[/error] {e.message}", style="bold")
        raise typer.Exit(1)
