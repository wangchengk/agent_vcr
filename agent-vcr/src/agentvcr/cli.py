"""CLI interface for AgentVCR."""

from __future__ import annotations

import click
from rich.console import Console

from agentvcr.tape import Tape
from agentvcr.player import play_file

console = Console()


@click.group()
@click.version_option()
def main():
    """📼 AgentVCR — Record, replay, and debug AI Agents."""
    pass


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--open", "open_browser", is_flag=True, help="Open in browser")
@click.option("--steps", default=None, help="Step range (e.g., 3-7)")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def replay(path: str, open_browser: bool, steps: str | None, verbose: bool) -> None:
    """📼 Replay a recorded session."""
    step_range = None
    if steps:
        parts = steps.split("-")
        start, end = int(parts[0]), int(parts[1])
        step_range = range(start, end + 1)

    play_file(path, open_browser=open_browser, steps=step_range)
    if verbose:
        tape = Tape.load(path)
        console.print(f"\n[dim]Full tape data: {tape.model_dump_json(indent=2)[:500]}...[/dim]")


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--format", "-f", default="html", type=click.Choice(["html", "markdown", "md", "json", "pytest"]))
@click.option("--output", "-o", default=None, help="Output file path")
def export(path: str, format: str, output: str | None) -> None:
    """📤 Export a recording to various formats."""
    tape = Tape.load(path)

    if output is None:
        suffix = {"html": ".html", "markdown": ".md", "md": ".md", "json": ".json", "pytest": ".py"}[format]
        output = path.rsplit(".", 1)[0] + suffix

    result = tape.export(output, format=format)
    console.print(f"✅ Exported to: [bold green]{result}[/bold green]")


@main.command()
@click.argument("path", type=click.Path(exists=True))
def info(path: str) -> None:
    """ℹ️ Show recording metadata."""
    tape = Tape.load(path)

    from rich.table import Table

    table = Table(title=f"📼 {tape.name or tape.session_id}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="bold")

    table.add_row("Session ID", tape.session_id)
    table.add_row("Name", tape.name or "—")
    table.add_row("Created", tape.created_at)
    table.add_row("Steps", str(tape.step_count))
    table.add_row("Tool Calls", str(len(tape.tool_calls)))
    table.add_row("Errors", str(len(tape.errors)))
    table.add_row("Duration", f"{tape.duration_ms / 1000:.1f}s")
    table.add_row("Total Tokens", str(tape.total_tokens))
    if tape.estimated_cost_usd > 0:
        table.add_row("Est. Cost", f"${tape.estimated_cost_usd:.4f}")

    console.print(table)


@main.command()
def list_recordings() -> None:
    """📋 List all recordings in the current directory."""
    from pathlib import Path

    recordings = list(Path("recordings").glob("*.json")) if Path("recordings").exists() else []

    if not recordings:
        console.print("[dim]No recordings found in ./recordings/[/dim]")
        return

    from rich.table import Table

    table = Table(title="📼 Recordings")
    table.add_column("File", style="cyan")
    table.add_column("Session", style="bold")
    table.add_column("Steps")
    table.add_column("Tokens")

    for r in recordings:
        try:
            tape = Tape.load(r)
            table.add_row(r.name, tape.name or tape.session_id, str(tape.step_count), str(tape.total_tokens))
        except Exception:
            table.add_row(r.name, "—", "—", "—")

    console.print(table)


if __name__ == "__main__":
    main()
