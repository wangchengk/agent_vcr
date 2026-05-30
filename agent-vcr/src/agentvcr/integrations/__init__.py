"""Framework integrations for AgentVCR."""

from agentvcr.integrations.openai import record_agent as record_openai
from agentvcr.integrations.anthropic import record_agent as record_anthropic
from agentvcr.integrations.langchain import record_chain as record_langchain
from agentvcr.integrations.crewai import record_crew as record_crewai

__all__ = ["record_openai", "record_anthropic", "record_langchain", "record_crewai"]
