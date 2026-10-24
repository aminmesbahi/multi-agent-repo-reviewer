"""Decision-model routing: a small typed classifier proposes, validation disposes.

A *decision model* answers fixed yes/no (or pick-one) questions with probabilities instead of
generating text. Here it is asked, per candidate agent, "does this request call for that
review?" which gives multi-label routing with one independent probability per agent.

Safety properties enforced in this file, independent of any backend:
  * the agent ids in a decision must be exactly the candidate allowlist (no more, no fewer)
  * every score must be a finite number in [0, 1]
  * any failure, malformed output or low confidence yields the conservative fallback:
    run every candidate agent. A router can therefore only ever add cost, never hide a review.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from repo_reviewer.models import AgentName, RoutingDecision
from repo_reviewer.routing.base import AGENT_DESCRIPTIONS, RoutingRequest

log = logging.getLogger("repo_reviewer.routing")

# ILLUSTRATIVE, NOT CALIBRATED. 0.5 is the natural boundary for an independent yes/no
# probability, nothing more. Choose a real value by running scripts/compare_routers.py
# --sweep against a labeled dataset and a real backend (see docs/routing.md).
DEFAULT_SELECT_THRESHOLD = 0.5


class RoutingFailure(Exception):
    """Base class for anything that makes a decision unusable."""


class BackendError(RoutingFailure):
    """The decision backend could not be reached or returned an error status."""


class MalformedDecision(RoutingFailure):
    """The backend answered, but not in the shape or range we require."""


class UnsupportedAgent(MalformedDecision):
    """The backend named an agent that is not in the allowlist."""


@dataclass(frozen=True)
class DecisionQuery:
    request_text: str
    facts: Mapping[str, int | bool]
    agents: Mapping[str, str]  # allowlisted agent id -> description


@dataclass(frozen=True)
class BackendOutput:
    """Raw, unvalidated backend answer: agent id -> probability that it should run."""

    scores: Mapping[str, object]
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    meta: dict[str, str] = field(default_factory=dict)


class DecisionBackend(Protocol):
    """Adapter to one decision-model inference API. Implementations own their wire format."""

    name: str

    def score(self, query: DecisionQuery) -> BackendOutput: ...


def validate_scores(raw: object, candidates: tuple[AgentName, ...]) -> dict[AgentName, float]:
    """Turn a raw backend mapping into typed scores or raise a RoutingFailure."""
    if not isinstance(raw, Mapping):
        raise MalformedDecision("scores must be a mapping of agent id to probability")
    allowed = {a.value for a in AgentName}
    unknown = sorted(str(k) for k in raw if str(k) not in allowed)
    if unknown:
        raise UnsupportedAgent(f"unsupported agent id(s): {unknown}")
    expected = {a.value for a in candidates}
    if set(map(str, raw)) != expected:
        raise MalformedDecision(
            f"expected scores for exactly {sorted(expected)}, got {sorted(map(str, raw))}"
        )
    scores: dict[AgentName, float] = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise MalformedDecision(f"score for '{key}' is not a number")
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise MalformedDecision(f"score for '{key}' is outside [0, 1]")
        scores[AgentName(str(key))] = float(value)
    return scores


class DecisionModelRouter:
    name = "decision"

    def __init__(
        self, backend: DecisionBackend, threshold: float = DEFAULT_SELECT_THRESHOLD
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.backend = backend
        self.threshold = threshold

    def route(self, request: RoutingRequest) -> RoutingDecision:
        started = time.perf_counter()
        query = DecisionQuery(
            request_text=request.bounded_text,
            facts=dict(request.facts),
            agents={a.value: AGENT_DESCRIPTIONS[a] for a in request.candidates},
        )
        output: BackendOutput | None = None
        try:
            output = self.backend.score(query)
            scores = validate_scores(output.scores, request.candidates)
        except RoutingFailure as exc:
            return self._fallback(request, started, output, f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - a routing bug must never abort the review
            return self._fallback(request, started, output, f"unexpected {type(exc).__name__}")

        selected = tuple(a for a in request.candidates if scores[a] >= self.threshold)
        if not selected:
            return self._fallback(
                request, started, output, f"low confidence: no score >= {self.threshold}", scores
            )
        return RoutingDecision(
            router=f"{self.name}:{self.backend.name}",
            source="decision_model",
            agents=selected,
            scores=scores,
            reason=f"scores >= {self.threshold} (illustrative threshold)",
            **self._usage(output, started),
        )

    def _fallback(
        self,
        request: RoutingRequest,
        started: float,
        output: BackendOutput | None,
        why: str,
        scores: dict[AgentName, float] | None = None,
    ) -> RoutingDecision:
        log.warning("decision routing fell back to all agents: %s", why)
        return RoutingDecision(
            router=f"{self.name}:{self.backend.name}",
            source="fallback",
            agents=request.candidates,
            scores=scores,
            reason="conservative fallback: running every applicable agent",
            fallback_reason=why,
            **self._usage(output, started),
        )

    @staticmethod
    def _usage(output: BackendOutput | None, started: float) -> dict[str, float | int | None]:
        return {
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "input_tokens": output.input_tokens if output else None,
            "output_tokens": output.output_tokens if output else None,
            "cost_usd": output.cost_usd if output else None,
        }
