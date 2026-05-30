"""Comprehensive tests for AgentVCR core models."""

import json
import tempfile
from pathlib import Path

import pytest

from agentvcr import Tape, Step, StepType, record, VCR


# ─────────────────────────────────────────────────────────
# TestStep
# ─────────────────────────────────────────────────────────

class TestStep:
    def test_step_creation_all_fields(self):
        step = Step(
            index=0, type=StepType.TOOL_CALL, timestamp_ms=1234,
            content="search", input={"q": "flights"}, output={"results": []},
            duration_ms=500, tokens=50, cost_usd=0.001, metadata={"tool": "search"}
        )
        assert step.index == 0
        assert step.type == StepType.TOOL_CALL
        assert step.input == {"q": "flights"}
        assert step.output == {"results": []}
        assert step.duration_ms == 500
        assert step.tokens == 50
        assert step.cost_usd == 0.001

    def test_step_summary_think(self):
        step = Step(index=0, type=StepType.THINK, timestamp_ms=0, content="I should search")
        assert "think" in step.summary
        assert "I should search" in step.summary

    def test_step_summary_tool_call(self):
        step = Step(index=1, type=StepType.TOOL_CALL, timestamp_ms=0, content="db_query", metadata={"tool": "db_query"})
        assert "db_query" in step.summary

    def test_step_summary_error(self):
        step = Step(index=2, type=StepType.ERROR, timestamp_ms=0, content="Connection timeout")
        assert "error" in step.summary
        assert "Connection timeout" in step.summary

    def test_step_summary_observe(self):
        step = Step(index=3, type=StepType.OBSERVE, timestamp_ms=0, content="", output={"status": 200})
        assert "observe" in step.summary

    def test_step_summary_truncates_long_content(self):
        long_content = "A" * 200
        step = Step(index=0, type=StepType.THINK, timestamp_ms=0, content=long_content)
        assert len(step.summary) < 100  # Truncated

    def test_step_metadata_default_empty(self):
        step = Step(index=0, type=StepType.THINK, timestamp_ms=0, content="hello")
        assert step.metadata == {}


# ─────────────────────────────────────────────────────────
# TestTape
# ─────────────────────────────────────────────────────────

class TestTape:
    def test_tape_defaults(self):
        tape = Tape()
        assert tape.session_id is not None
        assert tape.step_count == 0
        assert tape.total_tokens == 0
        assert tape.estimated_cost_usd == 0.0

    def test_tape_with_name(self):
        tape = Tape(name="my-session")
        assert tape.name == "my-session"

    def test_add_step_increments_index(self):
        tape = Tape(name="test")
        tape._start_time = 0
        s1 = tape.add_step(type=StepType.THINK, content="step1")
        s2 = tape.add_step(type=StepType.THINK, content="step2")
        assert s1.index == 0
        assert s2.index == 1

    def test_add_step_accumulates_tokens(self):
        tape = Tape(name="test")
        tape._start_time = 0
        tape.add_step(type=StepType.THINK, content="a", tokens=10)
        tape.add_step(type=StepType.THINK, content="b", tokens=20)
        assert tape.total_tokens == 30

    def test_add_step_accumulates_cost(self):
        tape = Tape(name="test")
        tape._start_time = 0
        tape.add_step(type=StepType.THINK, content="a", cost_usd=0.001)
        tape.add_step(type=StepType.THINK, content="b", cost_usd=0.002)
        assert tape.estimated_cost_usd == 0.003

    def test_add_step_timestamp_increases(self, monkeypatch):
        tape = Tape(name="test")
        # Mock perf_counter to return predictable values
        counter_vals = [100.0, 100.1, 100.2, 100.3]  # start + 2 add_step calls + 1 extra
        def mock_counter():
            return counter_vals.pop(0)
        monkeypatch.setattr("time.perf_counter", mock_counter)
        tape.start()
        tape.add_step(type=StepType.THINK, content="first")
        tape.add_step(type=StepType.THINK, content="second")
        # Second step should have ~100ms elapsed
        assert tape.steps[1].timestamp_ms >= 99

    def test_add_step_with_complex_input_output(self):
        tape = Tape(name="test")
        tape._start_time = 0
        complex_input = {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}
        complex_output = [{"id": 1, "score": 95.5}, {"id": 2, "score": 88.0}]
        tape.add_step(type=StepType.TOOL_CALL, content="rank_users", input=complex_input, output=complex_output)
        assert tape.steps[0].input["users"][0]["name"] == "Alice"
        assert tape.steps[0].output[0]["score"] == 95.5

    def test_start_stop_tape(self):
        tape = Tape(name="test")
        tape.start()
        assert tape._start_time is not None
        assert len(tape.steps) >= 1
        assert tape.steps[0].type == StepType.START
        initial_count = len(tape.steps)
        tape.stop()
        assert len(tape.steps) > initial_count
        assert tape.duration_ms > 0

    def test_tool_calls_returns_only_tool_calls(self):
        tape = Tape(name="test")
        tape._start_time = 0
        tape.add_step(type=StepType.THINK, content="a")
        tape.add_step(type=StepType.TOOL_CALL, content="t1", metadata={"tool": "t1"})
        tape.add_step(type=StepType.OBSERVE, content="b")
        tape.add_step(type=StepType.TOOL_CALL, content="t2", metadata={"tool": "t2"})
        calls = tape.tool_calls
        assert len(calls) == 2
        assert all(s.type == StepType.TOOL_CALL for s in calls)

    def test_errors_returns_only_errors(self):
        tape = Tape(name="test")
        tape._start_time = 0
        tape.add_step(type=StepType.THINK, content="ok")
        tape.add_step(type=StepType.ERROR, content="err1")
        tape.add_step(type=StepType.ERROR, content="err2")
        tape.add_step(type=StepType.THINK, content="recovered")
        errors = tape.errors
        assert len(errors) == 2
        assert all(s.type == StepType.ERROR for s in errors)

    def test_save_json_format(self, tmp_path):
        tape = Tape(name="json-test")
        tape._start_time = 0
        tape.add_step(type=StepType.THINK, content="hello", tokens=10)
        tape.duration_ms = 100
        path = tape.save(tmp_path / "tape.json")
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["name"] == "json-test"
        assert len(data["steps"]) == 1
        assert data["total_tokens"] == 10

    def test_load_json_roundtrip(self, tmp_path):
        tape = Tape(name="roundtrip")
        tape._start_time = 0
        tape.add_step(type=StepType.TOOL_CALL, content="test", input={"x": 1}, metadata={"tool": "test"})
        path = tmp_path / "roundtrip.json"
        tape.save(path)
        loaded = Tape.load(path)
        assert loaded.name == "roundtrip"
        assert loaded.step_count == 1
        assert loaded.steps[0].input["x"] == 1
        assert loaded.steps[0].metadata["tool"] == "test"

    def test_save_creates_parent_dirs(self, tmp_path):
        tape = Tape(name="nested")
        tape._start_time = 0
        path = tmp_path / "deeply" / "nested" / "dir" / "tape.json"
        result = tape.save(path)
        assert result.exists()

    def test_session_id_unique(self):
        t1 = Tape()
        t2 = Tape()
        assert t1.session_id != t2.session_id

    def test_created_at_is_iso_format(self):
        tape = Tape()
        assert "T" in tape.created_at
        assert "+" in tape.created_at or "Z" in tape.created_at

    def test_tape_model_validate(self):
        data = {
            "session_id": "abc123",
            "name": "validate-test",
            "steps": [
                {"index": 0, "type": "think", "timestamp_ms": 0, "content": "hello", "tokens": 5}
            ],
            "duration_ms": 100,
            "total_tokens": 5,
        }
        tape = Tape.model_validate(data)
        assert tape.session_id == "abc123"
        assert tape.step_count == 1

    def test_tape_to_pytest(self, tmp_path):
        tape = Tape(name="flight-search")
        tape._start_time = 0
        tape.add_step(type=StepType.THINK, content="thinking")
        tape.add_step(type=StepType.TOOL_CALL, content="search", metadata={"tool": "search"})
        tape.add_step(type=StepType.ERROR, content="failed")
        pytest_path = tmp_path / "test_flight_search.py"
        result = tape.to_pytest(pytest_path)
        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert "test_flight_search" in content
        assert "assert_step_contains" in content

    def test_tape_export_html(self, tmp_path):
        tape = Tape(name="html-export")
        tape._start_time = 0
        tape.add_step(type=StepType.THINK, content="hello")
        path = tape.export(tmp_path / "out.html", format="html")
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "<html" in content
        assert "AgentVCR" in content
        assert "hello" in content

    def test_tape_export_markdown(self, tmp_path):
        tape = Tape(name="md-export")
        tape._start_time = 0
        tape.add_step(type=StepType.THINK, content="hello world")
        path = tape.export(tmp_path / "out.md", format="markdown")
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "#" in content
        assert "hello world" in content

    def test_tape_export_json(self, tmp_path):
        tape = Tape(name="json-export")
        tape._start_time = 0
        tape.add_step(type=StepType.THINK, content="hello")
        path = tape.export(tmp_path / "out.json", format="json")
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["name"] == "json-export"

    def test_tape_export_unknown_format_raises(self, tmp_path):
        tape = Tape(name="bad-format")
        tape._start_time = 0
        with pytest.raises(ValueError, match="Unknown format"):
            tape.export(tmp_path / "out.xyz", format="xyz")


# ─────────────────────────────────────────────────────────
# TestAssertions
# ─────────────────────────────────────────────────────────

class TestAssertions:
    def test_assert_step_type_found(self):
        tape = Tape(name="test")
        tape._start_time = 0
        tape.add_step(type=StepType.THINK, content="a")
        tape.add_step(type=StepType.TOOL_CALL, content="b", metadata={"tool": "b"})
        found = tape.assert_step_type(StepType.TOOL_CALL)
        assert found.content == "b"

    def test_assert_step_type_by_string(self):
        tape = Tape(name="test")
        tape._start_time = 0
        tape.add_step(type=StepType.ERROR, content="boom")
        found = tape.assert_step_type("error")
        assert found.content == "boom"

    def test_assert_step_type_not_found_raises(self):
        tape = Tape(name="test")
        tape._start_time = 0
        tape.add_step(type=StepType.THINK, content="a")
        with pytest.raises(AssertionError, match="No step of type"):
            tape.assert_step_type(StepType.ERROR)

    def test_assert_step_type_wrong_index_raises(self):
        tape = Tape(name="test")
        tape._start_time = 0
        tape.add_step(type=StepType.THINK, content="a")
        with pytest.raises(AssertionError, match="is StepType.THINK"):
            tape.assert_step_type(StepType.TOOL_CALL, index=0)

    def test_assert_step_contains_found(self):
        tape = Tape(name="test")
        tape._start_time = 0
        tape.add_step(type=StepType.THINK, content="The quick brown fox")
        found = tape.assert_step_contains("fox")
        assert "fox" in found.content

    def test_assert_step_contains_case_insensitive(self):
        tape = Tape(name="test")
        tape._start_time = 0
        tape.add_step(type=StepType.THINK, content="Hello World")
        found = tape.assert_step_contains("hello")
        assert "Hello World" in found.content

    def test_assert_step_contains_with_type_filter(self):
        tape = Tape(name="test")
        tape._start_time = 0
        tape.add_step(type=StepType.THINK, content="search term")
        tape.add_step(type=StepType.TOOL_CALL, content="something else")
        found = tape.assert_step_contains("search", type=StepType.THINK)
        assert found.type == StepType.THINK

    def test_assert_step_contains_not_found_raises(self):
        tape = Tape(name="test")
        tape._start_time = 0
        tape.add_step(type=StepType.THINK, content="hello")
        with pytest.raises(AssertionError):
            tape.assert_step_contains("goodbye")


# ─────────────────────────────────────────────────────────
# TestVCR
# ─────────────────────────────────────────────────────────

class TestVCR:
    def test_vcr_creates_output_dir(self, tmp_path):
        vcr = VCR(output_dir=tmp_path / "recordings")
        assert vcr.output_dir.exists()

    def test_vcr_record_starts_tape(self, tmp_path):
        vcr = VCR(output_dir=tmp_path)
        tape = vcr.record("my-tape")
        assert tape.name == "my-tape"
        assert len(tape.steps) >= 1
        assert tape.steps[0].type == StepType.START

    def test_vcr_save_naming(self, tmp_path):
        vcr = VCR(output_dir=tmp_path)
        tape = vcr.record("named-tape")
        tape.stop()
        path = vcr.save(tape)
        assert path.name == "named-tape.json"

    def test_vcr_save_unnamed_uses_session_id(self, tmp_path):
        vcr = VCR(output_dir=tmp_path)
        tape = vcr.record()
        tape.stop()
        path = vcr.save(tape)
        assert tape.session_id in path.name


# ─────────────────────────────────────────────────────────
# TestRecordContext
# ─────────────────────────────────────────────────────────

class TestRecordContext:
    def test_context_saves_on_exit(self, tmp_path):
        with record("ctx-save", output_dir=tmp_path) as tape:
            tape.add_step(type=StepType.THINK, content="saving")

        saved_file = tmp_path / "ctx-save.json"
        assert saved_file.exists()

    def test_context_records_error_on_exception(self, tmp_path):
        try:
            with record("ctx-error", output_dir=tmp_path) as tape:
                tape.add_step(type=StepType.THINK, content="about to fail")
                raise ValueError("intentional failure")
        except ValueError:
            pass

        saved_file = tmp_path / "ctx-error.json"
        assert saved_file.exists()
        loaded = Tape.load(saved_file)
        # Should have recorded the error step
        assert any(s.type == StepType.ERROR for s in loaded.steps)


# ─────────────────────────────────────────────────────────
# TestRecordFunction
# ─────────────────────────────────────────────────────────

class TestRecordFunction:
    def test_record_function_decorator(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from agentvcr.recorder import record_function

        @record_function("decorated-func")
        def multiply(a: int, b: int) -> int:
            return a * b

        result = multiply(6, 7)
        assert result == 42

    def test_record_function_error_propagates(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from agentvcr.recorder import record_function

        @record_function("will-fail")
        def will_fail() -> None:
            raise RuntimeError("expected error")

        with pytest.raises(RuntimeError, match="expected error"):
            will_fail()


# ─────────────────────────────────────────────────────────
# TestEdgeCases
# ─────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_tape_with_zero_steps(self, tmp_path):
        tape = Tape(name="empty")
        assert tape.step_count == 0
        path = tape.save(tmp_path / "empty.json")
        loaded = Tape.load(path)
        assert loaded.step_count == 0

    def test_tape_with_unicode_content(self, tmp_path):
        tape = Tape(name="unicode-test")
        tape._start_time = 0
        tape.add_step(type=StepType.THINK, content="中文内容 🎉 emoji 内容")
        tape.add_step(type=StepType.ERROR, content="エラー 日本語")
        path = tape.save(tmp_path / "unicode.json")
        loaded = Tape.load(path)
        assert "中文内容" in loaded.steps[0].content
        assert "エラー" in loaded.steps[1].content

    def test_tape_with_none_output(self, tmp_path):
        tape = Tape(name="none-output")
        tape._start_time = 0
        tape.add_step(type=StepType.TOOL_CALL, content="test", input={"x": 1}, output=None)
        path = tape.save(tmp_path / "none_out.json")
        loaded = Tape.load(path)
        assert loaded.steps[0].output is None

    def test_tape_with_string_output(self, tmp_path):
        tape = Tape(name="str-output")
        tape._start_time = 0
        tape.add_step(type=StepType.OBSERVE, content="", output="plain string result")
        path = tape.save(tmp_path / "str_out.json")
        loaded = Tape.load(path)
        assert loaded.steps[0].output == "plain string result"

    def test_tape_cost_usd_zero(self):
        tape = Tape(name="zero-cost")
        tape._start_time = 0
        tape.add_step(type=StepType.THINK, content="no cost info")
        assert tape.estimated_cost_usd == 0.0

    def test_tape_duration_zero(self):
        tape = Tape(name="zero-duration")
        assert tape.duration_ms == 0.0

    def test_step_types_all_variants(self):
        for st in StepType:
            step = Step(index=0, type=st, timestamp_ms=0, content=f"testing {st.value}")
            assert step.type == st
            assert st.value in step.summary or step.content in step.summary
