"""Tests for framework integrations."""

import pytest

from agentvcr import Tape, Step, StepType


class TestOpenAIIntegration:
    def test_record_agent_returns_context(self):
        from agentvcr.integrations.openai import record_agent
        with record_agent("test-openai") as tape:
            tape.add_step(type=StepType.THINK, content="test")
        assert tape.step_count >= 2  # start + think + end

    def test_record_response_parses_text(self):
        from agentvcr.integrations.openai import record_response

        class FakeChoice:
            class FakeMessage:
                content = "The capital of France is Paris."
                tool_calls = None
            choice = None

        class FakeUsage:
            prompt_tokens = 10
            completion_tokens = 20
            total_tokens = 30

        class FakeResponse:
            choices = None
            usage = None

        # Build fake response
        msg = FakeChoice.FakeMessage()
        FakeChoice.choice = type("C", (), {"message": msg})()
        FakeResponse.choices = [FakeChoice.choice]
        FakeResponse.usage = FakeUsage()

        tape = Tape(name="test")
        tape._start_time = 0
        record_response(tape, FakeResponse())
        # Should have at least the text content step and usage step
        assert any("Paris" in s.content for s in tape.steps)


class TestAnthropicIntegration:
    def test_record_agent_returns_context(self):
        from agentvcr.integrations.anthropic import record_agent
        with record_agent("test-anthropic") as tape:
            tape.add_step(type=StepType.THINK, content="test")
        assert tape.step_count >= 2

    def test_record_response_parses_text_block(self):
        from agentvcr.integrations.anthropic import record_response

        class FakeBlock:
            type = "text"
            text = "Let me search for that."

        class FakeUsage:
            input_tokens = 15
            output_tokens = 25

        class FakeResponse:
            content = [FakeBlock()]
            usage = FakeUsage()

        tape = Tape(name="test")
        tape._start_time = 0
        record_response(tape, FakeResponse())
        assert any("search" in s.content for s in tape.steps)


class TestLangChainIntegration:
    def test_record_chain_returns_context(self):
        from agentvcr.integrations.langchain import record_chain
        with record_chain("test-langchain") as tape:
            tape.add_step(type=StepType.THINK, content="test")
        assert tape.step_count >= 2

    def test_record_invoke_parses_agent_result(self):
        from agentvcr.integrations.langchain import record_invoke

        class FakeAction:
            tool = "search"
            tool_input = {"query": "flights"}

        tape = Tape(name="test")
        tape._start_time = 0
        result = {
            "input": "find flights to Tokyo",
            "output": "Found 5 flights",
            "intermediate_steps": [
                (FakeAction(), "5 results found")
            ]
        }
        record_invoke(tape, result)
        assert any("find flights" in s.content for s in tape.steps)
        assert any(s.type == StepType.TOOL_CALL for s in tape.steps)


class TestCrewAIIntegration:
    def test_record_crew_returns_context(self):
        from agentvcr.integrations.crewai import record_crew
        with record_crew("test-crewai") as tape:
            tape.add_step(type=StepType.THINK, content="test")
        assert tape.step_count >= 2
