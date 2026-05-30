"""Recorder — the main recording engine."""

from __future__ import annotations

import functools
import time
from pathlib import Path
from typing import Any, Callable

from agentvcr.tape import StepType, Tape


class VCR:
    """AgentVCR main controller.

    Usage::

        vcr = VCR(output_dir="recordings")
        tape = vcr.record("my-session")
        # ... run your agent ...
        tape.stop()
        tape.save()
    """

    def __init__(self, output_dir: str | Path = "recordings"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def record(self, name: str = "", **metadata: Any) -> Tape:
        """Create a new tape and start recording."""
        tape = Tape(name=name, metadata=metadata)
        tape.start()
        return tape

    def save(self, tape: Tape) -> Path:
        """Save a tape to the output directory."""
        filename = f"{tape.name or tape.session_id}.json"
        return tape.save(self.output_dir / filename)


class _RecordContext:
    """Context manager for `with record(...) as tape:`."""

    def __init__(self, name: str = "", output_dir: str | Path = "recordings", **metadata: Any):
        self.vcr = VCR(output_dir=output_dir)
        self.tape = self.vcr.record(name, **metadata)

    def __enter__(self) -> Tape:
        return self.tape

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.tape.add_step(
                type=StepType.ERROR,
                content=f"{exc_type.__name__}: {exc_val}",
            )
        self.tape.stop()
        self.vcr.save(self.tape)


def record(name: str = "", output_dir: str | Path = "recordings", **metadata: Any) -> _RecordContext:
    """Create a recording context.

    Usage::

        with record("my-session") as tape:
            tape.add_step(StepType.THINK, content="Planning...")
            result = agent.run("hello")
    """
    return _RecordContext(name=name, output_dir=output_dir, **metadata)


def record_function(name: str = "", output_dir: str | Path = "recordings"):
    """Decorator to record a function as a tape.

    Usage::

        @record_function("my-agent")
        def run_agent(prompt: str):
            return agent.run(prompt)

        run_agent("hello")  # Automatically recorded
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            session_name = name or func.__name__
            with _RecordContext(name=session_name, output_dir=output_dir) as tape:
                tape.add_step(
                    type=StepType.THINK,
                    content=f"Calling {func.__name__}({args}, {kwargs})",
                )
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    duration = (time.perf_counter() - start) * 1000
                    tape.add_step(
                        type=StepType.OBSERVE,
                        content=f"Function returned successfully",
                        output={"result": str(result)[:500]},
                        duration_ms=duration,
                    )
                    return result
                except Exception as e:
                    duration = (time.perf_counter() - start) * 1000
                    tape.add_step(
                        type=StepType.ERROR,
                        content=str(e),
                        duration_ms=duration,
                    )
                    raise

        return wrapper

    return decorator
