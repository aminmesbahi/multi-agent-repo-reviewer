"""Deterministic, model-free router. This is the default.

With no request text it selects every applicable agent, which is exactly the behaviour of
the original planner. With request text it matches keywords; if nothing matches it falls back
to every applicable agent rather than guessing.
"""

from __future__ import annotations

import re

from repo_reviewer.models import AgentName, RoutingDecision
from repo_reviewer.routing.base import RoutingRequest

KEYWORDS: dict[AgentName, re.Pattern[str]] = {
    AgentName.CODE: re.compile(
        r"\b(code|bugs?|exceptions?|error[- ]handling|try/except|resources?|leaks?|complexity|"
        r"maintainab\w*|refactor\w*|source|functions?|crash\w*)\b",
        re.IGNORECASE,
    ),
    AgentName.DEPENDENCIES: re.compile(
        r"\b(dependenc\w*|pinn?(ed|ing)?|requirements?|pyproject|versions?|lock\s?files?|"
        r"reproducib\w*|packages?)\b",
        re.IGNORECASE,
    ),
    AgentName.DOCUMENTATION: re.compile(
        r"\b(readme|docs?|documentation|install\w*|usage|prerequisites?|instructions?)\b",
        re.IGNORECASE,
    ),
}


class RuleBasedRouter:
    name = "deterministic"

    def route(self, request: RoutingRequest) -> RoutingDecision:
        text = request.bounded_text
        if not text:
            return self._all(request, "no request text; running every applicable agent")
        matched = tuple(a for a in request.candidates if KEYWORDS[a].search(text))
        if not matched:
            return self._all(request, "no keyword matched; running every applicable agent")
        return RoutingDecision(
            router=self.name,
            source="rules",
            agents=matched,
            reason="keyword match: " + ", ".join(a.value for a in matched),
        )

    def _all(self, request: RoutingRequest, reason: str) -> RoutingDecision:
        return RoutingDecision(
            router=self.name, source="rules", agents=request.candidates, reason=reason
        )
