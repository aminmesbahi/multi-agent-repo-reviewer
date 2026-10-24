"""Agent routing: rules, decision-model router, validation, fallback, adapter wire format."""

from __future__ import annotations

import json

import httpx
import pytest

from repo_reviewer.config import ConfigError, DecisionSettings, decision_settings
from repo_reviewer.models import AgentName, RoutingDecision
from repo_reviewer.orchestrator.planner import ProjectFacts, plan_review
from repo_reviewer.routing import DecisionModelRouter, RoutingRequest, RuleBasedRouter
from repo_reviewer.routing.backends import (
    MockDecisionBackend,
    StaticDecisionBackend,
    SystemOneBackend,
)
from repo_reviewer.routing.decision import (
    BackendError,
    MalformedDecision,
    UnsupportedAgent,
    validate_scores,
)

ALL = tuple(AgentName)
CODE, DEPS, DOCS = AgentName.CODE, AgentName.DEPENDENCIES, AgentName.DOCUMENTATION


def req(text: str | None, candidates=ALL) -> RoutingRequest:
    return RoutingRequest(text, candidates, {"python_files": 3, "manifests": 2, "has_readme": True})


def scores(code=0.1, deps=0.1, docs=0.1) -> dict[str, float]:
    return {"code": code, "dependencies": deps, "documentation": docs}


def router(raw=None, error=None, threshold=0.5) -> DecisionModelRouter:
    return DecisionModelRouter(StaticDecisionBackend(raw, error), threshold)


# -- deterministic router (default) ---------------------------------------------------------


def test_rules_without_text_select_everything():
    decision = RuleBasedRouter().route(req(None))
    assert decision.agents == ALL and decision.source == "rules"


def test_rules_keywords_select_subsets():
    r = RuleBasedRouter()
    assert r.route(req("please review the README")).agents == (DOCS,)
    assert set(r.route(req("check pinned dependencies and error handling")).agents) == {CODE, DEPS}


def test_rules_fall_back_to_all_when_nothing_matches():
    assert RuleBasedRouter().route(req("is it any good?")).agents == ALL


def test_rules_respect_candidates():
    assert RuleBasedRouter().route(req("readme and code", candidates=(DOCS,))).agents == (DOCS,)


# -- decision router: success paths ---------------------------------------------------------


def test_multi_label_selection():
    decision = router(scores(code=0.9, deps=0.2, docs=0.8)).route(req("x"))
    assert decision.agents == (CODE, DOCS) and decision.source == "decision_model"
    assert decision.scores[CODE] == 0.9


def test_single_label_selection():
    assert router(scores(deps=0.97)).route(req("x")).agents == (DEPS,)


def test_threshold_is_inclusive_and_configurable():
    assert router(scores(code=0.5), threshold=0.5).route(req("x")).agents == (CODE,)
    assert router(scores(code=0.5, docs=0.3), threshold=0.25).route(req("x")).agents == (CODE, DOCS)


def test_only_candidates_are_queried_and_returned():
    decision = router({"code": 0.9, "documentation": 0.9}).route(req("x", (CODE, DOCS)))
    assert decision.agents == (CODE, DOCS)


# -- decision router: conservative fallback -------------------------------------------------


def assert_fallback(decision: RoutingDecision, needle: str, candidates=ALL):
    assert decision.source == "fallback"
    assert decision.agents == candidates  # runs everything: never hides a review
    assert needle in decision.fallback_reason


def test_low_confidence_everywhere_falls_back():
    assert_fallback(
        router(scores(0.3, 0.2, 0.49)).route(req("review my project")), "low confidence"
    )


def test_ambiguous_request_with_mock_backend_falls_back():
    decision = DecisionModelRouter(MockDecisionBackend()).route(req("take a look at this please"))
    assert_fallback(decision, "low confidence")


def test_mock_backend_multi_agent_request():
    decision = DecisionModelRouter(MockDecisionBackend()).route(
        req("update the readme install steps and pin the requirements")
    )
    assert decision.source == "decision_model" and set(decision.agents) == {DOCS, DEPS}


@pytest.mark.parametrize(
    ("raw", "needle"),
    [
        (
            {"code": 0.9, "dependencies": 0.9, "documentation": 0.9, "shell": 0.9},
            "UnsupportedAgent",
        ),
        ({"code": 0.9, "dependencies": 0.9, "documentation": 0.9, "Code": 0.9}, "UnsupportedAgent"),
        ({"code": 0.9, "dependencies": 0.9}, "MalformedDecision"),  # agent missing
        ({"code": 1.2, "dependencies": 0.1, "documentation": 0.1}, "outside [0, 1]"),
        ({"code": -0.1, "dependencies": 0.1, "documentation": 0.1}, "outside [0, 1]"),
        ({"code": float("nan"), "dependencies": 0.1, "documentation": 0.1}, "outside [0, 1]"),
        ({"code": "high", "dependencies": 0.1, "documentation": 0.1}, "not a number"),
        ({"code": True, "dependencies": 0.1, "documentation": 0.1}, "not a number"),
        ({"code": None, "dependencies": 0.1, "documentation": 0.1}, "not a number"),
        (["code", "docs"], "must be a mapping"),
        ("code,documentation", "must be a mapping"),
        (None, "must be a mapping"),
    ],
)
def test_malformed_decisions_are_rejected_with_fallback(raw, needle):
    assert_fallback(router(raw).route(req("x")), needle)


def test_agent_not_among_candidates_is_rejected():
    decision = router(scores()).route(req("x", (CODE, DOCS)))  # backend also scored 'dependencies'
    assert_fallback(decision, "exactly", (CODE, DOCS))


@pytest.mark.parametrize(
    "error",
    [
        BackendError("endpoint down"),
        MalformedDecision("bad json"),
        RuntimeError("bug"),
        TimeoutError(),
    ],
)
def test_routing_failures_fall_back(error):
    decision = router(error=error).route(req("x"))
    assert decision.source == "fallback" and decision.agents == ALL


def test_unexpected_error_text_is_not_leaked_into_the_decision():
    decision = router(error=RuntimeError("secret request text")).route(req("x"))
    assert "secret" not in decision.fallback_reason


def test_validate_scores_direct():
    assert validate_scores(scores(0, 1, 0.5), ALL)[DEPS] == 1.0
    with pytest.raises(UnsupportedAgent):
        validate_scores({"rm": 1}, ALL)


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        DecisionModelRouter(MockDecisionBackend(), 0.0)


def test_routing_decision_model_validates_itself():
    with pytest.raises(ValueError):
        RoutingDecision(router="x", source="rules", agents=(), reason="r")
    with pytest.raises(ValueError):
        RoutingDecision(router="x", source="rules", agents=(CODE, CODE), reason="r")
    with pytest.raises(ValueError):
        RoutingDecision(router="x", source="rules", agents=("shell",), reason="r")


# -- planner integration --------------------------------------------------------------------


FACTS = ProjectFacts(python_files=3, manifests=1, has_readme=True)


def test_planner_router_cannot_add_inapplicable_agents():
    facts = ProjectFacts(python_files=0, manifests=1, has_readme=True)
    plan = plan_review(facts, None, "x", router({"dependencies": 0.9, "documentation": 0.9}))
    assert plan.selected == (DEPS, DOCS) and plan.skipped["code"] == "no Python files found"


def test_planner_marks_unselected_agents_as_skipped():
    plan = plan_review(FACTS, None, "x", router(scores(code=0.9)))
    assert plan.selected == (CODE,)
    assert plan.skipped == {
        "dependencies": "not selected by decision:static",
        "documentation": "not selected by decision:static",
    }


def test_planner_explicit_agents_bypass_the_router():
    class Exploding:
        name = "boom"

        def route(self, request):
            raise AssertionError("router must not be consulted")

    plan = plan_review(FACTS, [DOCS], "x", Exploding())
    assert plan.selected == (DOCS,) and plan.routing.source == "cli"


def test_planner_defensive_when_router_returns_foreign_agents():
    class Foreign:
        name = "foreign"

        def route(self, request):
            return RoutingDecision(router="foreign", source="rules", agents=(CODE,), reason="r")

    facts = ProjectFacts(python_files=0, manifests=1, has_readme=True)
    plan = plan_review(facts, None, None, Foreign())
    assert plan.selected == (DEPS, DOCS) and plan.routing.source == "fallback"


# -- configuration / privacy guards ---------------------------------------------------------


def test_remote_decision_endpoint_refused_unless_opted_in():
    env = {"LITELLM_MASTER_KEY": "sk-x", "DECISION_API_BASE": "https://decisions.example.com"}
    with pytest.raises(ConfigError, match="not local"):
        decision_settings(env)
    assert decision_settings({**env, "DECISION_ALLOW_REMOTE": "1"}).allow_remote


@pytest.mark.parametrize(
    "base",
    [
        "http://127.0.0.1:4000",
        "http://localhost:4000",
        "http://[::1]:4000",
        "http://host.docker.internal:4000",
    ],
)
def test_local_decision_endpoints_accepted(base):
    assert decision_settings({"LITELLM_MASTER_KEY": "sk-x", "DECISION_API_BASE": base})


def test_decision_settings_need_a_key():
    with pytest.raises(ConfigError, match="LITELLM_MASTER_KEY"):
        decision_settings({})


# -- SystemOneBackend: wire format (contract tests against the documented shape) ------------

SETTINGS = DecisionSettings("http://127.0.0.1:4000", "sk-test", "router-decision")


def systemone(handler) -> SystemOneBackend:
    return SystemOneBackend(SETTINGS, httpx.Client(transport=httpx.MockTransport(handler)))


def good_response(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    answers = {
        name: {"type": "noul", "noul": 0.9 if name == "documentation" else 0.05}
        for name in body["questions"]
    }
    return httpx.Response(
        200,
        json={
            "model": body["model"],
            "answers": answers,
            "usage": {"input_tokens": 120, "output_tokens": 9},
        },
        headers={"x-litellm-response-cost": "0.000012"},
    )


def test_systemone_request_shape_and_privacy():
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["auth"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content)
        return good_response(request)

    decision = DecisionModelRouter(systemone(handler)).route(req("improve the install docs"))
    assert captured["url"] == "http://127.0.0.1:4000/v1/systemone"
    assert captured["auth"] == "Bearer sk-test"
    body = captured["body"]
    assert body["model"] == "router-decision"
    assert set(body["questions"]) == {"code", "dependencies", "documentation"}
    assert {q["type"] for q in body["questions"].values()} == {"noul"}
    assert body["questions"]["code"]["criteria"].keys() == {"true", "false"}
    assert "improve the install docs" in body["state"]
    # only the request text and coarse facts are sent: no paths, no file contents
    assert "python_files=3" in body["state"] and ".py" not in body["state"]
    assert decision.agents == (DOCS,)
    assert (decision.input_tokens, decision.output_tokens, decision.cost_usd) == (120, 9, 0.000012)


@pytest.mark.parametrize(
    ("response", "needle"),
    [
        (httpx.Response(500, json={}), "HTTP 500"),
        (httpx.Response(401, json={}), "HTTP 401"),
        (httpx.Response(200, text="<html>"), "did not return JSON"),
        (httpx.Response(200, json={"answers": []}), "no 'answers'"),
        (httpx.Response(200, json={}), "no 'answers'"),
        (
            httpx.Response(
                200,
                json={
                    "answers": {
                        "code": {"type": "choice", "choice": "a"},
                        "dependencies": {"type": "noul", "noul": 0.1},
                        "documentation": {"type": "noul", "noul": 0.1},
                    }
                },
            ),
            "not a number",
        ),
        (
            httpx.Response(
                200,
                json={
                    "answers": {
                        "code": {"type": "noul", "noul": 0.1},
                        "dependencies": {"type": "noul", "noul": 0.1},
                        "documentation": {"type": "noul", "noul": 0.1},
                        "exec": {"type": "noul", "noul": 1.0},
                    }
                },
            ),
            "UnsupportedAgent",
        ),
    ],
)
def test_systemone_bad_responses_fall_back(response, needle):
    decision = DecisionModelRouter(systemone(lambda r: response)).route(req("x"))
    assert decision.source == "fallback" and decision.agents == ALL
    assert needle in decision.fallback_reason


def test_systemone_network_failure_falls_back():
    def handler(request):
        raise httpx.ConnectError("refused")

    decision = DecisionModelRouter(systemone(handler)).route(req("x"))
    assert decision.source == "fallback" and "cannot reach" in decision.fallback_reason
