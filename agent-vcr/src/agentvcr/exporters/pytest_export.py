"""Pytest exporter — generate test files from recordings."""

from __future__ import annotations

from pathlib import Path

from agentvcr.tape import StepType, Tape


def export_pytest(tape: Tape, output_path: str | Path) -> Path:
    """Export a tape as a pytest test file.

    Generates a test that replays the recorded steps and asserts
    the agent follows the expected flow.
    """
    output_path = Path(output_path)

    test_name = f"test_{tape.name.replace('-', '_') or tape.session_id}"
    fixture_name = f"tape_{tape.name.replace('-', '_') or tape.session_id}"

    # Build assertions
    assertions = []
    for step in tape.steps:
        if step.type == StepType.TOOL_CALL:
            tool = step.metadata.get("tool", "unknown")
            assertions.append(
                f'    tape.assert_step_contains("{tool}", type="tool_call")'
            )
        elif step.type == StepType.ERROR:
            assertions.append(
                f"    errors = tape.errors\n"
                f"    assert len(errors) <= {len(tape.errors)}  # known errors"
            )

    # Build the test file
    content = f'''"""Auto-generated test from AgentVCR recording: {tape.name or tape.session_id}."""

import pytest
from agentvcr import Tape


# Load the recorded tape
TAPE_PATH = "recordings/{tape.name or tape.session_id}.json"


class Test{tape.name.replace("-", "").title() if tape.name else tape.session_id.title()}:
    """Tests generated from recording: {tape.name or tape.session_id}"""

    @pytest.fixture
    def {fixture_name}(self) -> Tape:
        return Tape.load(TAPE_PATH)

    def {test_name}(self, {fixture_name}):
        """Replay and validate agent session."""
        tape = {fixture_name}
        assert tape.step_count == {tape.step_count}
        assert len(tape.tool_calls) == {len(tape.tool_calls)}

        # Step-by-step assertions
{chr(10).join(assertions) if assertions else "        # No specific assertions generated"}

    def test_no_unexpected_errors(self, {fixture_name}):
        """Verify no unexpected errors in recording."""
        tape = {fixture_name}
        for error in tape.errors:
            # Add known error patterns here
            pass
'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path
