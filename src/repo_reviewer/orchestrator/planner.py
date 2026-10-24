"""Deterministic planner: decides which workers run. Never calls a model itself.

Order of authority:
  1. Applicability: an agent with nothing to look at (no .py files, no manifests) never runs.
  2. An explicit ``--agents`` list wins over any router.
  3. Otherwise the configured ``AgentRouter`` picks among the applicable agents. A router
     (rules or decision model) can only choose from, never extend, the applicable set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from repo_reviewer.models import AgentName, Limits, RoutingDecision
from repo_reviewer.orchestrator.state import Plan
from repo_reviewer.routing.base import AgentRouter, RoutingRequest
from repo_reviewer.tools.filesystem import Sandbox, find_readme
from repo_reviewer.tools.manifests import find_manifests

log = logging.getLogger("repo_reviewer.planner")


class NoApplicableAgents(Exception):
    """Nothing to run. The message is meant to be shown to the user."""


@dataclass(frozen=True)
class ProjectFacts:
    python_files: int
    manifests: int
    has_readme: bool

    def summary(self) -> dict[str, int | bool]:
        """Coarse counts only: this is the only project information a router may see."""
        return {
            "python_files": self.python_files,
            "manifests": self.manifests,
            "has_readme": self.has_readme,
        }


def scan_facts(repo_root: str, limits: Limits) -> ProjectFacts:
    sandbox = Sandbox(repo_root, limits)
    files, _ = sandbox.iter_files(".")
    return ProjectFacts(
        python_files=sum(1 for path, _size in files if path.endswith(".py")),
        manifests=len(find_manifests(sandbox)),
        has_readme=find_readme(sandbox) is not None,
    )


def inapplicable_reasons(facts: ProjectFacts) -> dict[AgentName, str]:
    reasons: dict[AgentName, str] = {}
    if facts.python_files == 0:
        reasons[AgentName.CODE] = "no Python files found"
    if facts.manifests == 0:
        reasons[AgentName.DEPENDENCIES] = "no pyproject.toml or requirements*.txt found"
    return reasons  # documentation always applies: a missing README is itself reviewable


def plan_review(
    facts: ProjectFacts,
    requested: list[AgentName] | None,
    request_text: str | None,
    router: AgentRouter,
) -> Plan:
    reasons = inapplicable_reasons(facts)
    applicable = tuple(a for a in AgentName if a not in reasons)
    skipped: dict[str, str] = {a.value: why for a, why in reasons.items()}

    if requested:
        wanted = [a for a in AgentName if a in set(requested)]
        selected = tuple(a for a in wanted if a in applicable)
        for agent in wanted:
            if agent in reasons:
                log.warning("agent=%s skipped: %s", agent.value, reasons[agent])
        if not selected:
            raise NoApplicableAgents(
                "none of the requested agents can run: "
                + "; ".join(f"{a.value} ({reasons[a]})" for a in wanted if a in reasons)
            )
        routing = RoutingDecision(
            router="cli", source="cli", agents=selected, reason="explicit --agents"
        )
        skipped = {k: v for k, v in skipped.items() if AgentName(k) in wanted}
        return Plan(selected, skipped, routing)

    if not applicable:
        raise NoApplicableAgents("no agent has anything to review in this repository")
    decision = router.route(RoutingRequest(request_text, applicable, facts.summary()))
    selected = tuple(a for a in applicable if a in decision.agents)
    if not selected:  # defensive: a router must not return agents outside the candidate set
        log.warning("router %s returned no applicable agent; using all", decision.router)
        selected = applicable
        decision = decision.model_copy(
            update={
                "source": "fallback",
                "agents": applicable,
                "fallback_reason": "router selected no applicable agent",
            }
        )
    for agent in applicable:
        if agent not in selected:
            skipped[agent.value] = f"not selected by {decision.router}"
    return Plan(selected, skipped, decision)
