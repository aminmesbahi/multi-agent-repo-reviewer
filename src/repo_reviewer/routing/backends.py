"""Decision-model backends. Interchangeable behind ``DecisionBackend``.

* ``MockDecisionBackend``   - deterministic lexical scorer for CI/demos. NOT a model.
* ``StaticDecisionBackend`` - returns a fixed answer or raises; used to test failure handling.
* ``SystemOneBackend``      - EXPERIMENTAL adapter for the LiteLLM Proxy ``/v1/systemone`` route.

The wire format used by ``SystemOneBackend`` was checked against the LiteLLM 1.106.3 source
and docs (question type ``noul``, answer ``{"type": "noul", "noul": <probability>}``), and the
chain adapter -> live proxy -> fake upstream was exercised. No real decision model was run
against it, so real-model behaviour is unverified (see docs/routing.md).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping

import httpx

from repo_reviewer.config import DecisionSettings
from repo_reviewer.routing.decision import (
    BackendError,
    BackendOutput,
    DecisionQuery,
    MalformedDecision,
)

log = logging.getLogger("repo_reviewer.routing")

_STEMS: dict[str, tuple[str, ...]] = {
    "code": (
        "code",
        "bug",
        "exception",
        "error",
        "except",
        "resource",
        "leak",
        "complex",
        "maintain",
        "refactor",
        "function",
        "source",
        "crash",
        "handl",
        "python",
    ),
    "dependencies": (
        "depend",
        "pin",
        "requirement",
        "pyproject",
        "version",
        "lock",
        "reproduc",
        "package",
        "manifest",
    ),
    "documentation": (
        "readme",
        "doc",
        "install",
        "usage",
        "prerequisit",
        "instruction",
        "guide",
        "onboard",
        "quickstart",
        "explain",
    ),
}


class MockDecisionBackend:
    """Deterministic stand-in: scores by counting stem hits. Not a model, no network."""

    name = "mock"

    def score(self, query: DecisionQuery) -> BackendOutput:
        tokens = re.findall(r"[a-z]+", query.request_text.lower())
        scores: dict[str, object] = {}
        for agent in query.agents:
            hits = sum(
                1 for stem in _STEMS.get(agent, ()) if any(t.startswith(stem) for t in tokens)
            )
            scores[agent] = 0.05 if hits == 0 else min(0.95, 0.55 + 0.2 * (hits - 1))
        return BackendOutput(scores=scores, input_tokens=0, output_tokens=0, cost_usd=0.0)


class StaticDecisionBackend:
    """Returns ``scores`` as-is (possibly malformed on purpose) or raises ``error``."""

    name = "static"

    def __init__(self, scores: object = None, error: Exception | None = None) -> None:
        self._scores = scores
        self._error = error

    def score(self, query: DecisionQuery) -> BackendOutput:
        if self._error is not None:
            raise self._error
        return BackendOutput(scores=self._scores)  # type: ignore[arg-type]


class SystemOneBackend:
    """EXPERIMENTAL. One ``noul`` (probability-of-yes) question per agent via LiteLLM.

    Request:  POST {api_base}/v1/systemone
              {"model": ..., "state": "<request + coarse facts>",
               "questions": {"<agent id>": {"type": "noul", "instructions": ..., "criteria": ...}}}
    Response: {"answers": {"<agent id>": {"type": "noul", "noul": 0.93}}, "usage": {...}}

    The state contains only the user's request text and counts/booleans, never repository
    contents. ``DecisionSettings`` refuses non-local endpoints unless explicitly allowed.
    """

    name = "systemone-experimental"

    def __init__(self, settings: DecisionSettings, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self._client = client or httpx.Client(timeout=settings.timeout_s)

    @staticmethod
    def build_body(model: str, query: DecisionQuery) -> dict[str, object]:
        facts = "; ".join(f"{k}={v}" for k, v in sorted(query.facts.items()))
        return {
            "model": model,
            "state": f"User request: {query.request_text}\nProject facts: {facts}",
            "questions": {
                agent: {
                    "type": "noul",
                    "instructions": (
                        f"Does this request call for the '{agent}' review? "
                        f"That reviewer {description}."
                    ),
                    "criteria": {
                        "true": f"The request asks for or implies the {agent} review.",
                        "false": f"The request is unrelated to the {agent} review.",
                    },
                }
                for agent, description in query.agents.items()
            },
        }

    def score(self, query: DecisionQuery) -> BackendOutput:
        url = f"{self.settings.api_base}/v1/systemone"
        log.warning("EXPERIMENTAL decision backend: POST %s model=%s", url, self.settings.model)
        try:
            response = self._client.post(
                url,
                json=self.build_body(self.settings.model, query),
                headers={"Authorization": f"Bearer {self.settings.api_key}"},
            )
        except httpx.HTTPError as exc:
            raise BackendError(f"cannot reach decision endpoint ({type(exc).__name__})") from exc
        if response.status_code != 200:
            raise BackendError(f"decision endpoint returned HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError as exc:
            raise MalformedDecision("decision endpoint did not return JSON") from exc
        answers = data.get("answers") if isinstance(data, dict) else None
        if not isinstance(answers, Mapping):
            raise MalformedDecision("response has no 'answers' mapping")
        scores: dict[str, object] = {}
        for agent, answer in answers.items():
            ok = isinstance(answer, Mapping) and answer.get("type") == "noul"
            scores[str(agent)] = answer.get("noul") if ok else None
        usage = data.get("usage") if isinstance(data.get("usage"), Mapping) else {}
        return BackendOutput(
            scores=scores,
            input_tokens=_as_int(usage.get("input_tokens")),
            output_tokens=_as_int(usage.get("output_tokens")),
            cost_usd=_as_float(response.headers.get("x-litellm-response-cost")),
        )


def _as_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _as_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
