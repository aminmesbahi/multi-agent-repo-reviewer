"""Graph state and its reducer.

LangGraph runs the ``Send``-dispatched workers concurrently in one superstep. Several of them
write ``results`` at the same time, which is only legal because that key has a reducer.
Every other key is written by exactly one node.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, TypedDict

from repo_reviewer.models import (
    AgentName,
    Limits,
    Mode,
    ReviewSummary,
    RoutingDecision,
    WorkerResult,
)

_ORDER = {name: index for index, name in enumerate(AgentName)}


def merge_results(
    left: list[WorkerResult] | None, right: list[WorkerResult] | None
) -> list[WorkerResult]:
    """Reducer for ``ReviewState.results``.

    Order-independent (the same set of results merges to the same list regardless of which
    worker finished first) and idempotent per agent (a later result for the same agent
    replaces the earlier one), so the aggregate is deterministic under parallelism.
    """
    by_agent: dict[AgentName, WorkerResult] = {}
    for result in [*(left or []), *(right or [])]:
        by_agent[result.agent] = result
    return sorted(by_agent.values(), key=lambda r: _ORDER[r.agent])


@dataclass(frozen=True)
class Plan:
    """Output of the planner node: who runs, who was skipped and why."""

    selected: tuple[AgentName, ...]
    skipped: dict[str, str]
    routing: RoutingDecision


class ReviewState(TypedDict, total=False):
    # inputs (set once by the caller)
    repo_root: str
    mode: Mode
    limits: Limits
    deadline: float
    requested_agents: list[AgentName] | None
    request_text: str | None
    # written by the planner node only
    plan: Plan
    # written concurrently by worker nodes -> needs the reducer
    results: Annotated[list[WorkerResult], merge_results]
    # written by the finalize node only
    summary: ReviewSummary
    report: str
