"""Registry of the three specialised agents."""

from repo_reviewer.agents import code_reviewer, dependencies, documentation
from repo_reviewer.agents.base import AgentSpec, run_agent
from repo_reviewer.models import AgentName

AGENTS: dict[AgentName, AgentSpec] = {
    AgentName.CODE: code_reviewer.SPEC,
    AgentName.DEPENDENCIES: dependencies.SPEC,
    AgentName.DOCUMENTATION: documentation.SPEC,
}

__all__ = ["AGENTS", "AgentSpec", "run_agent"]
