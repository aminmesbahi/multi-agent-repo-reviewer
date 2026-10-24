"""The LangGraph workflow. No LLM call happens in this file.

    START -> plan --(conditional edge: one Send per selected agent)--> worker (parallel)
                                                                          |
                                                                       finalize -> END

* ``plan``     deterministic planner (routing is delegated to an ``AgentRouter``).
* ``dispatch`` the conditional edge function; returns ``Send("worker", WorkerInput)`` per agent.
* ``worker``   runs one agent's tool loop with an isolated context; never raises.
* ``finalize`` verifies, deduplicates, sorts, renders. Uses only the reducer-merged results.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from repo_reviewer.agents import AGENTS, AgentSpec, run_agent
from repo_reviewer.gateway import ChatClient
from repo_reviewer.models import (
    AgentName,
    Limits,
    Mode,
    ReviewSummary,
    WorkerInput,
    WorkerResult,
)
from repo_reviewer.orchestrator.planner import plan_review, scan_facts
from repo_reviewer.orchestrator.state import ReviewState
from repo_reviewer.orchestrator.validation import finalize_findings
from repo_reviewer.reporting.markdown import render_markdown
from repo_reviewer.routing.base import AgentRouter
from repo_reviewer.routing.rules import RuleBasedRouter

log = logging.getLogger("repo_reviewer.graph")


def build_graph(
    client: ChatClient,
    router: AgentRouter | None = None,
    agents: dict[AgentName, AgentSpec] | None = None,
) -> CompiledStateGraph:
    """Compile the review graph. ``client`` and ``router`` are injected (no globals)."""
    router = router or RuleBasedRouter()
    specs = agents or AGENTS

    def plan_node(state: ReviewState) -> dict[str, object]:
        facts = scan_facts(state["repo_root"], state["limits"])
        plan = plan_review(facts, state.get("requested_agents"), state.get("request_text"), router)
        log.info("plan agents=%s router=%s", [a.value for a in plan.selected], plan.routing.router)
        return {"plan": plan}

    def dispatch(state: ReviewState) -> list[Send]:
        return [
            Send(
                "worker",
                WorkerInput(
                    agent=agent,
                    repo_root=state["repo_root"],
                    limits=state["limits"],
                    deadline=state["deadline"],
                ),
            )
            for agent in state["plan"].selected
        ]

    def worker_node(work: WorkerInput) -> dict[str, list[WorkerResult]]:
        try:
            result = run_agent(specs[work.agent], work, client)
        except Exception as exc:  # noqa: BLE001 - last line of defence for failure isolation
            log.exception("worker %s crashed outside the agent loop", work.agent.value)
            result = WorkerResult(
                agent=work.agent, status="failed", error=f"{type(exc).__name__}: {exc}"[:500]
            )
        return {"results": [result]}

    def finalize_node(state: ReviewState) -> dict[str, object]:
        results = state.get("results", [])
        findings, rejected, removed = finalize_findings(
            results, state["repo_root"], state["limits"]
        )
        plan = state["plan"]
        summary = ReviewSummary(
            repo=state["repo_root"],
            mode=state["mode"],
            routing=plan.routing,
            skipped_agents=plan.skipped,
            workers=results,
            findings=findings,
            rejected_findings=rejected,
            duplicates_removed=removed,
        )
        return {"summary": summary, "report": render_markdown(summary)}

    graph = StateGraph(ReviewState)
    graph.add_node("plan", plan_node)
    graph.add_node("worker", worker_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", dispatch, ["worker"])
    graph.add_edge("worker", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


@dataclass(frozen=True)
class ReviewResult:
    summary: ReviewSummary
    report: str


def run_review(
    repo_root: str,
    *,
    mode: Mode,
    client: ChatClient,
    router: AgentRouter | None = None,
    requested_agents: list[AgentName] | None = None,
    request_text: str | None = None,
    limits: Limits | None = None,
) -> ReviewResult:
    """Convenience wrapper: build the graph, run it once, return summary + Markdown."""
    limits = limits or Limits()
    graph = build_graph(client, router)
    final = graph.invoke(
        {
            "repo_root": repo_root,
            "mode": mode,
            "limits": limits,
            "deadline": time.monotonic() + limits.total_timeout_s,
            "requested_agents": requested_agents,
            "request_text": request_text,
            "results": [],
        }
    )
    return ReviewResult(summary=final["summary"], report=final["report"])
