"""Anthropic integration — record Claude agent sessions."""

from __future__ import annotations

from typing import Any

from agentvcr.recorder import _RecordContext, record
from agentvcr.tape import StepType, Tape


def record_agent(name: str = "anthropic-agent", **kwargs: Any) -> _RecordContext:
    """Create a recording context for Anthropic agents.

    Usage::

        from agentvcr.integrations.anthropic import record_agent

        with record_agent("my-claude-session") as tape:
            response = client.messages.create(...)
            record_response(tape, response)
    """
    return record(name, **kwargs)


def record_response(tape: Tape, response: Any) -> None:
    """Record an Anthropic Message response as steps on the tape."""
    for block in response.content:
        if block.type == "text":
            tape.add_step(
                type=StepType.THINK,
                content=block.text[:500],
                tokens=response.usage.output_tokens if response.usage else 0,
            )
        elif block.type == "tool_use":
            tape.add_step(
                type=StepType.TOOL_CALL,
                content=block.name,
                input=block.input if isinstance(block.input, dict) else {"input": str(block.input)},
                metadata={"tool": block.name, "tool_use_id": block.id},
            )

    if response.usage:
        tape.add_step(
            type=StepType.OBSERVE,
            content="API call completed",
            metadata={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            tokens=response.usage.input_tokens + response.usage.output_tokens,
        )
