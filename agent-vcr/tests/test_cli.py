"""Tests for AgentVCR CLI commands."""

import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from agentvcr import Tape, Step, StepType, record


class TestCLI:
    """Test the agentvcr CLI commands."""

    @pytest.fixture
    def runner(self):
        from agentvcr.cli import main
        return CliRunner(mix_stderr=False)

    @pytest.fixture
    def sample_tape_path(self, tmp_path):
        tape = Tape(name="cli-test")
        tape._start_time = 0
        tape.add_step(type=StepType.THINK, content="Planning my trip", tokens=20)
        tape.add_step(type=StepType.TOOL_CALL, content="search", input={"q": "flights"}, metadata={"tool": "search"})
        tape.add_step(type=StepType.OBSERVE, content="Found 5 results", tokens=50)
        tape.duration_ms = 5000
        tape.total_tokens = 70
        path = tmp_path / "cli-test.json"
        tape.save(path)
        return path

    def test_cli_version(self, runner):
        from agentvcr.cli import main
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1" in result.stdout

    def test_cli_replay(self, runner, sample_tape_path):
        from agentvcr.cli import main
        result = runner.invoke(main, ["replay", str(sample_tape_path)])
        assert result.exit_code == 0
        assert "cli-test" in result.stdout
        assert "3 steps" in result.stdout

    def test_cli_replay_with_steps_filter(self, runner, sample_tape_path):
        from agentvcr.cli import main
        result = runner.invoke(main, ["replay", str(sample_tape_path), "--steps", "0-1"])
        assert result.exit_code == 0

    def test_cli_export_html(self, runner, sample_tape_path, tmp_path):
        from agentvcr.cli import main
        out = tmp_path / "exported.html"
        result = runner.invoke(main, ["export", str(sample_tape_path), "-f", "html", "-o", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "html" in content.lower()
        assert "cli-test" in content

    def test_cli_export_markdown(self, runner, sample_tape_path, tmp_path):
        from agentvcr.cli import main
        out = tmp_path / "exported.md"
        result = runner.invoke(main, ["export", str(sample_tape_path), "-f", "markdown", "-o", str(out)])
        assert result.exit_code == 0
        assert out.exists()

    def test_cli_export_pytest(self, runner, sample_tape_path, tmp_path):
        from agentvcr.cli import main
        out = tmp_path / "test_output.py"
        result = runner.invoke(main, ["export", str(sample_tape_path), "-f", "pytest", "-o", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "def test" in content

    def test_cli_info(self, runner, sample_tape_path):
        from agentvcr.cli import main
        result = runner.invoke(main, ["info", str(sample_tape_path)])
        assert result.exit_code == 0
        assert "cli-test" in result.stdout
        assert "Steps" in result.stdout
        assert "3" in result.stdout

    def test_cli_info_shows_cost(self, runner, tmp_path):
        from agentvcr.cli import main
        tape = Tape(name="cost-test")
        tape._start_time = 0
        tape.add_step(type=StepType.THINK, content="test", cost_usd=0.015)
        path = tmp_path / "cost.json"
        tape.save(path)
        result = runner.invoke(main, ["info", str(path)])
        assert result.exit_code == 0
        assert "0.015" in result.stdout

    def test_cli_replay_nonexistent_file(self, runner):
        from agentvcr.cli import main
        result = runner.invoke(main, ["replay", "nonexistent.json"])
        assert result.exit_code != 0

    def test_cli_export_unknown_format(self, runner, sample_tape_path):
        from agentvcr.cli import main
        result = runner.invoke(main, ["export", str(sample_tape_path), "-f", "unknown_format"])
        assert result.exit_code != 0
