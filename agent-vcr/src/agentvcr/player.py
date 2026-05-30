"""Player — replay tapes in terminal or browser."""

from __future__ import annotations

import webbrowser
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agentvcr.tape import Step, StepType, Tape

console = Console()

# Emoji map for step types
STEP_EMOJI = {
    StepType.THINK: "🧠",
    StepType.TOOL_CALL: "🔧",
    StepType.OBSERVE: "👁️",
    StepType.ERROR: "❌",
    StepType.DECISION: "🔀",
    StepType.RETRY: "🔄",
    StepType.CUSTOM: "📌",
    StepType.START: "▶️",
    StepType.END: "⏹️",
}


class Player:
    """Replay a recorded tape.

    Usage::

        tape = Tape.load("recordings/my-session.json")
        Player(tape).play()
    """

    def __init__(self, tape: Tape):
        self.tape = tape

    def play(self, steps: range | None = None, verbose: bool = False) -> None:
        """Play the tape in the terminal with Rich formatting."""
        tape = self.tape

        # Header
        console.print()
        console.print(
            Panel.fit(
                f"📼 Playing: [bold]{tape.name or tape.session_id}[/bold]",
                subtitle=f"{tape.step_count} steps · {tape.duration_ms / 1000:.1f}s · {tape.total_tokens} tokens",
                border_style="magenta",
            )
        )

        # Timeline
        step_list = tape.steps
        if steps is not None:
            step_list = [s for s in step_list if s.index in steps]

        for step in step_list:
            self._render_step(step, verbose)

        # Summary
        console.print()
        summary = Table(show_header=False, box=None)
        summary.add_column(style="dim")
        summary.add_column(style="bold")
        summary.add_row("Total Steps", str(tape.step_count))
        summary.add_row("Tool Calls", str(len(tape.tool_calls)))
        summary.add_row("Errors", str(len(tape.errors)))
        summary.add_row("Duration", f"{tape.duration_ms / 1000:.1f}s")
        summary.add_row("Tokens", str(tape.total_tokens))
        if tape.estimated_cost_usd > 0:
            summary.add_row("Est. Cost", f"${tape.estimated_cost_usd:.4f}")
        console.print(Panel(summary, title="📊 Session Summary", border_style="blue"))

    def _render_step(self, step: Step, verbose: bool = False) -> None:
        """Render a single step."""
        emoji = STEP_EMOJI.get(step.type, "•")
        timestamp = f"{step.timestamp_ms / 1000:.1f}s"

        # Step header
        header = Text()
        header.append(f"  {emoji} ", style="bold")
        header.append(f"Step {step.index + 1}/{self.tape.step_count}", style="cyan")
        header.append(f"  [{timestamp}]", style="dim")
        header.append(f"  {step.type.value}", style="bold magenta")

        console.print(header)

        # Content
        if step.content:
            content = step.content[:200] + "..." if len(step.content) > 200 else step.content
            console.print(f"    {content}")

        # Tool call details
        if step.type == StepType.TOOL_CALL and verbose:
            if step.input:
                console.print(f"    [dim]Input:[/dim] {step.input}")
            if step.output:
                output_str = str(step.output)[:150]
                console.print(f"    [dim]Output:[/dim] {output_str}")

        # Error details
        if step.type == StepType.ERROR:
            console.print(f"    [red]{step.content}[/red]")

        # Duration & tokens
        details = []
        if step.duration_ms is not None:
            details.append(f"{step.duration_ms:.0f}ms")
        if step.tokens > 0:
            details.append(f"{step.tokens} tokens")
        if details:
            console.print(f"    [dim]{' · '.join(details)}[/dim]")

    def open_in_browser(self, port: int = 8742) -> None:
        """Open an interactive timeline in the browser (WIP)."""
        # TODO: Build a local web server with the timeline viewer
        console.print("[yellow]Browser viewer coming soon![/yellow]")
        console.print(f"For now, export to HTML: agentvcr export {self.tape.name}.json --format html")


def play_file(path: str | Path, open_browser: bool = False, steps: range | None = None) -> None:
    """Load and play a tape file."""
    tape = Tape.load(path)
    player = Player(tape)
    if open_browser:
        player.open_in_browser()
    else:
        player.play(steps=steps)
