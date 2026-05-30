"""CrewAI integration — record CrewAI agent sessions."""

from __future__ import annotations

from typing import Any

from agentvcr.recorder import _RecordContext, record
from agentvcr.tape import StepType, Tape


def record_crew(name: str = "crewai-crew", **kwargs: Any) -> _RecordContext:
    """Create a recording context for CrewAI crews.

    Usage::

        from agentvcr.integrations.crewai import record_crew

        with record_crew("my-crew") as tape:
            result = crew.kickoff()
            record_crew_result(tape, result)
    """
    return record(name, **kwargs)


def record_crew_result(tape: Tape, result: Any) -> None:
    """Record a CrewAI crew kickoff result."""
    if hasattr(result, "tasks_output"):
        for task_output in result.tasks_output:
            tape.add_step(
                type=StepType.THINK,
                content=f"Task: {getattr(task_output, 'description', '')[:200]}",
                output={"result": str(getattr(task_output, 'raw', ''))[:300]},
            )
    tape.add_step(type=StepType.OBSERVE, content=str(result)[:500])
