"""OpenAI integration — record OpenAI agent sessions."""

from __future__ import annotations

import time
from typing import Any

from agentvcr.recorder import _RecordContext, record
from agentvcr.tape import StepType, Tape


def record_agent(name: str = "openai-agent", **kwargs: Any) -> _RecordContext:
    """Create a recording context for OpenAI agents.

    Usage::

        from agentvcr.integrations.openai import record_agent

        with record_agent("my-gpt4-session") as tape:
            response = client.chat.completions.create(...)
            _record_response(tape, response)
    """
    return record(name, **kwargs)


def record_response(tape: Tape, response: Any) -> None:
    """Record an OpenAI ChatCompletion response as steps on the tape.

    Automatically extracts:
    - Tool calls as TOOL_CALL steps
    - Text content as THINK steps
    - Token usage
    - Latency
    """
    if not response.choices:
        tape.add_step(type=StepType.OBSERVE, content="Empty response from API")
        return

    choice = response.choices[0]
    message = choice.message

    # Record text content (agent "thinking")
    if message.content:
        tape.add_step(
            type=StepType.THINK,
            content=message.content[:500],
            tokens=response.usage.completion_tokens if response.usage else 0,
        )

    # Record tool calls
    if message.tool_calls:
        for tc in message.tool_calls:
            tape.add_step(
                type=StepType.TOOL_CALL,
                content=tc.function.name,
                input={"arguments": tc.function.arguments},
                metadata={"tool": tc.function.name, "tool_call_id": tc.id},
                tokens=0,
            )

    # Record usage
    if response.usage:
        tape.add_step(
            type=StepType.OBSERVE,
            content=f"API call completed",
            metadata={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            tokens=response.usage.total_tokens,
        )
