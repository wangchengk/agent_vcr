"""LangChain integration — record LangChain/LangGraph agent sessions."""

from __future__ import annotations

from typing import Any

from agentvcr.recorder import _RecordContext, record
from agentvcr.tape import StepType, Tape


def record_chain(name: str = "langchain-agent", **kwargs: Any) -> _RecordContext:
    """Create a recording context for LangChain agents.

    Usage::

        from agentvcr.integrations.langchain import record_chain

        with record_chain("my-langchain-session") as tape:
            result = agent.invoke({"input": "hello"})
            record_invoke(tape, result)
    """
    return record(name, **kwargs)


def record_invoke(tape: Tape, result: Any) -> None:
    """Record a LangChain agent invoke result."""
    if isinstance(result, dict):
        # Standard agent output
        if "input" in result:
            tape.add_step(type=StepType.THINK, content=f"Input: {str(result['input'])[:200]}")
        if "output" in result:
            tape.add_step(type=StepType.OBSERVE, content=str(result["output"])[:500])
        if "intermediate_steps" in result:
            for action, observation in result["intermediate_steps"]:
                tape.add_step(
                    type=StepType.TOOL_CALL,
                    content=str(action.tool),
                    input=action.tool_input if isinstance(action.tool_input, dict) else {"input": str(action.tool_input)},
                    output={"result": str(observation)[:300]},
                    metadata={"tool": action.tool},
                )
    else:
        tape.add_step(type=StepType.OBSERVE, content=str(result)[:500])
