"""The real LangGraph graph: planning, Send fan-out, concurrency, reducer, failure isolation."""

from __future__ import annotations

import json
import threading

import pytest
from conftest import make_repo

from repo_reviewer.gateway import GatewayError, ModelReply
from repo_reviewer.models import AgentName, Mode, Severity, WorkerResult
from repo_reviewer.orchestrator import build_graph, run_review
from repo_reviewer.orchestrator.planner import NoApplicableAgents
from repo_reviewer.orchestrator.state import merge_results
from repo_reviewer.testing.mock_model import MockModel


class AgentSwitch:
    """Delegates to MockModel but lets a test sabotage chosen agents."""

    def __init__(self, broken: dict[str, object] | None = None) -> None:
        self.broken = broken or {}
        self.inner = MockModel()
        self.threads: set[int] = set()

    def complete(self, messages, tools, *, timeout_s=None):
        agent = messages[0]["content"].split("\n", 1)[0].removeprefix("AGENT: ")
        self.threads.add(threading.get_ident())
        sabotage = self.broken.get(agent)
        if isinstance(sabotage, Exception):
            raise sabotage
        if isinstance(sabotage, str):
            return ModelReply(content=sabotage)
        return self.inner.complete(messages, tools, timeout_s=timeout_s)


@pytest.fixture
def repo(tmp_path):
    return make_repo(
        tmp_path / "r",
        {
            "pyproject.toml": '[project]\nname = "demo"\nrequires-python = ">=3.11"\n'
            'dependencies = ["requests"]\n',
            "demo.py": "def load(p):\n    return open(p).read()\n",
            "README.md": "# demo\n",
        },
    )


def run(repo, client, **kw):
    return run_review(str(repo), mode=Mode.MOCK, client=client, **kw)


def test_all_applicable_agents_run_through_the_graph(repo):
    result = run(repo, AgentSwitch())
    assert [w.agent for w in result.summary.workers] == list(AgentName)
    assert {f.agent for f in result.summary.findings} == set(AgentName)
    assert result.summary.routing.source == "rules"


def test_requested_agents_limit_the_fan_out(repo):
    result = run(repo, AgentSwitch(), requested_agents=[AgentName.DOCUMENTATION])
    assert [w.agent for w in result.summary.workers] == [AgentName.DOCUMENTATION]
    assert result.summary.routing.source == "cli"


def test_conditional_routing_skips_inapplicable_agents(tmp_path):
    repo = make_repo(tmp_path / "docs_only", {"README.md": "# only docs\n"})
    result = run(repo, AgentSwitch())
    assert [w.agent for w in result.summary.workers] == [AgentName.DOCUMENTATION]
    assert set(result.summary.skipped_agents) == {"code", "dependencies"}


def test_nothing_to_run_is_an_error(tmp_path):
    repo = make_repo(tmp_path / "empty", {"README.md": "# x\n"})
    with pytest.raises(NoApplicableAgents):
        run(repo, AgentSwitch(), requested_agents=[AgentName.CODE])


def test_dispatch_emits_one_send_per_selected_agent(repo):
    graph = build_graph(AgentSwitch())
    events = list(
        graph.stream(
            {
                "repo_root": str(repo),
                "mode": Mode.MOCK,
                "limits": __import__("repo_reviewer.models", fromlist=["Limits"]).Limits(),
                "deadline": __import__("time").monotonic() + 60,
                "results": [],
            },
            stream_mode="updates",
        )
    )
    nodes = [next(iter(e)) for e in events]
    assert nodes[0] == "plan" and nodes[-1] == "finalize"
    assert nodes.count("worker") == 3  # one Send per agent -> three worker invocations


def test_workers_really_run_in_parallel(repo):
    """A 3-party barrier only releases if all three workers are inside a model call at once."""
    barrier = threading.Barrier(3, timeout=10)

    class Rendezvous(AgentSwitch):
        first = True

        def complete(self, messages, tools, *, timeout_s=None):
            if not any(m["role"] == "tool" for m in messages):
                barrier.wait()
            return super().complete(messages, tools, timeout_s=timeout_s)

    client = Rendezvous()
    result = run(repo, client)
    assert all(w.status == "ok" for w in result.summary.workers)
    assert len(client.threads) >= 2


def test_parallel_results_are_aggregated_through_the_reducer(repo):
    result = run(repo, AgentSwitch())
    assert sorted(w.agent.value for w in result.summary.workers) == [
        "code",
        "dependencies",
        "documentation",
    ]
    assert len(result.summary.findings) == sum(len(w.findings) for w in result.summary.workers)


def test_failed_worker_does_not_destroy_other_results(repo):
    client = AgentSwitch({"dependencies": GatewayError("cannot reach the LiteLLM Proxy")})
    result = run(repo, client)
    by_agent = {w.agent: w for w in result.summary.workers}
    assert by_agent[AgentName.DEPENDENCIES].status == "failed"
    assert (
        by_agent[AgentName.CODE].status == "ok" and by_agent[AgentName.DOCUMENTATION].status == "ok"
    )
    assert {f.agent for f in result.summary.findings} == {AgentName.CODE, AgentName.DOCUMENTATION}
    assert "FAILED" in result.report and "partial" in result.report


def test_unexpected_exception_type_is_also_isolated(repo):
    result = run(repo, AgentSwitch({"code": RuntimeError("boom")}))
    assert {w.agent: w.status for w in result.summary.workers}[AgentName.CODE] == "failed"
    assert any(f.agent is AgentName.DOCUMENTATION for f in result.summary.findings)


def test_invalid_structured_output_from_one_worker_is_isolated(repo):
    result = run(repo, AgentSwitch({"documentation": "definitely not json"}))
    failed = result.summary.failed_workers
    assert [w.agent for w in failed] == [AgentName.DOCUMENTATION]
    assert result.summary.findings  # others survived


def test_hallucinated_findings_are_dropped_in_finalize(repo):
    fake = {
        "findings": [
            {
                "file": "does_not_exist.py",
                "line": 1,
                "severity": "high",
                "title": "Ghost bug",
                "description": "invented",
                "recommendation": "n/a",
                "evidence": "x",
            },
            {
                "file": "demo.py",
                "line": 999,
                "severity": "high",
                "title": "Ghost line",
                "description": "invented",
                "recommendation": "n/a",
                "evidence": None,
            },
            {
                "file": "demo.py",
                "line": 2,
                "severity": "high",
                "title": "Wrong quote",
                "description": "invented",
                "recommendation": "n/a",
                "evidence": "eval(user_input)",
            },
        ]
    }
    result = run(repo, AgentSwitch({"code": json.dumps(fake)}), requested_agents=[AgentName.CODE])
    assert result.summary.findings == []
    assert result.summary.rejected_findings == 3
    assert "discarded" in result.report


def test_reducer_is_order_independent_and_idempotent():
    def wr(agent, status="ok"):
        return WorkerResult(agent=agent, status=status)

    a, b, c = wr(AgentName.CODE), wr(AgentName.DEPENDENCIES), wr(AgentName.DOCUMENTATION)
    assert merge_results([a], [b, c]) == merge_results([c, b], [a])
    assert merge_results(merge_results([a], [b]), [b]) == merge_results([a], [b])
    assert merge_results(None, None) == []


def test_reducer_later_result_for_same_agent_wins():
    old = WorkerResult(agent=AgentName.CODE, status="failed", error="x")
    new = WorkerResult(agent=AgentName.CODE, status="ok")
    assert merge_results([old], [new])[0].status == "ok"


def test_graph_state_has_reducer_on_results_only():
    from typing import get_type_hints

    from repo_reviewer.orchestrator.state import ReviewState

    hints = get_type_hints(ReviewState, include_extras=True)
    annotated = {k for k, v in hints.items() if hasattr(v, "__metadata__")}
    assert annotated == {"results"}


def test_findings_are_sorted_deterministically(repo):
    first = run(repo, AgentSwitch()).summary.findings
    second = run(repo, AgentSwitch()).summary.findings
    assert first == second
    ranks = [f.severity.rank for f in first]
    assert ranks == sorted(ranks, reverse=True)
    assert all(isinstance(f.severity, Severity) for f in first)


def test_without_a_reducer_concurrent_worker_writes_are_rejected_by_langgraph():
    """Why `results` needs `merge_results`: LangGraph refuses parallel writes to a plain key."""
    from typing import TypedDict

    from langgraph.errors import InvalidUpdateError
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Send

    class Plain(TypedDict, total=False):
        results: list

    graph = StateGraph(Plain)
    graph.add_node("plan", lambda state: {})
    graph.add_node("worker", lambda state: {"results": [state["n"]]})
    graph.add_edge(START, "plan")
    graph.add_conditional_edges(
        "plan", lambda s: [Send("worker", {"n": i}) for i in range(3)], ["worker"]
    )
    graph.add_edge("worker", END)
    with pytest.raises(InvalidUpdateError):
        graph.compile().invoke({"results": []})
