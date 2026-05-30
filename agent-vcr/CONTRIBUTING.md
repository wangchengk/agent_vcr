# 🤝 Contributing to AgentVCR

Thanks for wanting to help debug the debugger! 📼

## Quick Start

1. Fork the repo
2. Create a branch: `git checkout -b my-feature`
3. Make changes & commit: `git commit -m "Add my feature"`
4. Push: `git push origin my-feature`
5. Open a Pull Request

## Adding a New Integration

This is the highest-impact contribution! Create a file in `src/agentvcr/integrations/`:

```python
from agentvcr.recorder import _RecordContext, record
from agentvcr.tape import StepType, Tape

def record_agent(name: str = "myframework-agent", **kwargs) -> _RecordContext:
    """Create a recording context for MyFramework agents."""
    return record(name, **kwargs)

def record_response(tape: Tape, response: Any) -> None:
    """Record a MyFramework response as steps on the tape."""
    # Parse your framework's response format
    # Add THINK, TOOL_CALL, OBSERVE, ERROR steps as appropriate
    tape.add_step(type=StepType.THINK, content="agent reasoning...")
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Code Style

- Python 3.10+
- Type hints required
- Use `ruff` for formatting
- Keep it simple — this is a debugging tool, it should be debuggable itself

## Reporting Issues

Found a bug? [Open an issue](../../issues/new) with:
- What you expected
- What happened
- The recording JSON if possible (sanitize any sensitive data first!)

## License

By contributing, you agree your code will be licensed under MIT.
