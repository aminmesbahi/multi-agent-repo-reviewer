"""The routing interface: *which agents run* (not which model answers; that is LiteLLM's job).

Three ways to decide, all behind ``AgentRouter``:
  * deterministic rules   -> routing/rules.py     (default, no model)
  * decision model        -> routing/decision.py  (small typed classifier, optional)
  * generative LLM router -> not implemented here; see docs/routing.md for why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from repo_reviewer.models import AgentName, RoutingDecision

AGENT_DESCRIPTIONS: dict[AgentName, str] = {
    AgentName.CODE: (
        "reviews Python source for error handling, resource management and maintainability"
    ),
    AgentName.DEPENDENCIES: (
        "inspects pyproject.toml and requirements files for unpinned or conflicting dependencies"
    ),
    AgentName.DOCUMENTATION: (
        "checks README installation, prerequisites and usage instructions against project metadata"
    ),
}

MAX_REQUEST_CHARS = 1_000


@dataclass(frozen=True)
class RoutingRequest:
    """Everything a router may see. Deliberately contains no repository contents or paths:
    only the user's request text and coarse counts/booleans about the project."""

    text: str | None
    candidates: tuple[AgentName, ...]
    facts: dict[str, int | bool] = field(default_factory=dict)

    @property
    def bounded_text(self) -> str:
        return (self.text or "").strip()[:MAX_REQUEST_CHARS]


class AgentRouter(Protocol):
    name: str

    def route(self, request: RoutingRequest) -> RoutingDecision:
        """Choose a non-empty subset of ``request.candidates``. Must not raise."""
        ...
