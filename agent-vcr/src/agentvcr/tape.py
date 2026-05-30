"""Tape & Step data models — the core recording format."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class StepType(str, Enum):
    """Types of agent steps that can be recorded."""

    THINK = "think"           # Agent reasoning / inner monologue
    TOOL_CALL = "tool_call"   # Agent calling a tool/API
    OBSERVE = "observe"       # Agent observing tool output
    ERROR = "error"           # Error occurred
    DECISION = "decision"     # Key decision point
    RETRY = "retry"           # Retry after failure
    CUSTOM = "custom"         # User-defined step type
    START = "start"           # Session start marker
    END = "end"               # Session end marker


class Step(BaseModel):
    """A single recorded step in an agent session."""

    index: int
    type: StepType
    timestamp_ms: float
    content: str = ""
    input: dict[str, Any] | None = None
    output: Any = None
    duration_ms: float | None = None
    tokens: int = 0
    cost_usd: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def summary(self) -> str:
        """One-line summary of this step."""
        if self.type == StepType.TOOL_CALL:
            tool = self.metadata.get("tool", "unknown")
            return f"tool_call({tool}): {self.content[:60]}"
        if self.type == StepType.THINK:
            return f"think: {self.content[:80]}"
        if self.type == StepType.ERROR:
            return f"error: {self.content[:80]}"
        if self.type == StepType.OBSERVE:
            return f"observe: {str(self.output)[:60] if self.output else ''}"
        return f"{self.type.value}: {self.content[:80]}"


class Tape(BaseModel):
    """A recorded agent session — the full tape."""

    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = ""
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    steps: list[Step] = Field(default_factory=list)
    duration_ms: float = 0.0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    # --- Recording state (not serialized) ---
    _start_time: float | None = None
    _start_tokens: int = 0

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def tool_calls(self) -> list[Step]:
        return [s for s in self.steps if s.type == StepType.TOOL_CALL]

    @property
    def errors(self) -> list[Step]:
        return [s for s in self.steps if s.type == StepType.ERROR]

    def add_step(
        self,
        type: StepType,
        content: str = "",
        input: dict[str, Any] | None = None,
        output: dict[str, Any] | str | None = None,
        duration_ms: float | None = None,
        tokens: int = 0,
        cost_usd: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Step:
        """Add a step to the tape."""
        step = Step(
            index=len(self.steps),
            type=type,
            timestamp_ms=self._elapsed_ms(),
            content=content,
            input=input,
            output=output,
            duration_ms=duration_ms,
            tokens=tokens,
            cost_usd=cost_usd,
            metadata=metadata or {},
        )
        self.steps.append(step)
        self.total_tokens += tokens
        if cost_usd is not None:
            self.estimated_cost_usd += cost_usd
        return step

    def _elapsed_ms(self) -> float:
        """Milliseconds since tape start."""
        import time

        if self._start_time is None:
            return 0.0
        return (time.perf_counter() - self._start_time) * 1000

    def start(self) -> None:
        """Start recording."""
        import time

        self._start_time = time.perf_counter()
        self.add_step(type=StepType.START, content=f"Session: {self.name or self.session_id}")

    def stop(self) -> None:
        """Stop recording."""
        if self._start_time is not None:
            import time

            self.duration_ms = (time.perf_counter() - self._start_time) * 1000
        self.add_step(type=StepType.END, content="Session ended")

    # --- Assertions for testing ---

    def assert_step_type(self, step_type: str | StepType, index: int | None = None) -> Step:
        """Assert a step of given type exists. Returns it."""
        st = StepType(step_type) if isinstance(step_type, str) else step_type
        if index is not None:
            step = self.steps[index]
            assert step.type == st, f"Step {index} is {step.type}, expected {st}"
            return step
        for step in self.steps:
            if step.type == st:
                return step
        raise AssertionError(f"No step of type {st} found in tape")

    def assert_step_contains(self, text: str, type: str | StepType | None = None) -> Step:
        """Assert a step contains text. Returns it."""
        for step in self.steps:
            if type is not None:
                st = StepType(type) if isinstance(type, str) else type
                if step.type != st:
                    continue
            if text.lower() in step.content.lower():
                return step
        raise AssertionError(f"No step contains '{text}'")

    # --- Serialization ---

    def save(self, path: str | Path) -> Path:
        """Save tape to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Use orjson if available, else stdlib json
        try:
            import orjson

            data = orjson.dumps(self.model_dump(), option=orjson.OPT_INDENT_2)
            path.write_bytes(data)
        except ImportError:
            path.write_text(
                json.dumps(self.model_dump(), indent=2, ensure_ascii=False, default=str)
            )
        return path

    @classmethod
    def load(cls, path: str | Path) -> Tape:
        """Load tape from JSON file."""
        path = Path(path)
        try:
            import orjson

            data = orjson.loads(path.read_bytes())
        except ImportError:
            data = json.loads(path.read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def to_pytest(self, output_path: str | Path) -> Path:
        """Export tape as a pytest file."""
        from agentvcr.exporters.pytest_export import export_pytest

        return export_pytest(self, output_path)

    def export(self, path: str | Path, format: str = "html") -> Path:
        """Export tape to various formats."""
        from agentvcr.exporters import export

        return export(self, path, format)
