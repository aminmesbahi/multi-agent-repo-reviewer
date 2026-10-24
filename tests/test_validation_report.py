"""Finding verification, deduplication, sorting, and Markdown rendering."""

from __future__ import annotations

import pytest
from conftest import make_repo

from repo_reviewer.models import (
    AgentName,
    Finding,
    Limits,
    Mode,
    ReviewSummary,
    RoutingDecision,
    Severity,
    WorkerResult,
)
from repo_reviewer.orchestrator.validation import deduplicate, finalize_findings, verify_finding
from repo_reviewer.reporting.markdown import render_markdown
from repo_reviewer.tools.filesystem import Sandbox


def finding(**kw) -> Finding:
    base = dict(
        agent=AgentName.CODE,
        file="a.py",
        line=2,
        severity=Severity.MEDIUM,
        title="Problem",
        description="desc",
        recommendation="fix",
        evidence=None,
    )
    return Finding(**{**base, **kw})


@pytest.fixture
def repo(tmp_path):
    return make_repo(tmp_path / "r", {"a.py": "import os\nopen('x')\n\n\nprint(1)\n"})


@pytest.fixture
def sandbox(repo):
    return Sandbox(repo, Limits(max_files=100))


def test_verify_levels(sandbox):
    assert verify_finding(finding(evidence="open('x')"), sandbox).verification == "quote"
    assert verify_finding(finding(), sandbox).verification == "line"
    assert verify_finding(finding(line=None), sandbox).verification == "file"


def test_verify_rejects_unverifiable_claims(sandbox):
    assert verify_finding(finding(file="missing.py"), sandbox) is None
    assert verify_finding(finding(line=500), sandbox) is None
    assert verify_finding(finding(evidence="eval(x)"), sandbox) is None
    assert verify_finding(finding(file="../escape.py"), sandbox) is None
    assert verify_finding(finding(file=".env"), sandbox) is None


def test_evidence_window_is_local(sandbox):
    # 'print(1)' is on line 5; citing line 1 is outside the +-2 window
    assert verify_finding(finding(line=1, evidence="print(1)"), sandbox) is None
    assert verify_finding(finding(line=4, evidence="print(1)"), sandbox) is not None


def test_whitespace_differences_in_evidence_are_tolerated(sandbox):
    assert verify_finding(
        finding(evidence="  open( 'x' )".replace("( ", "(").replace(" )", ")")), sandbox
    )


def test_deduplication_merges_same_location_and_title():
    a = finding(agent=AgentName.CODE, severity=Severity.LOW)
    b = finding(agent=AgentName.DOCUMENTATION, severity=Severity.HIGH, title="  problem ")
    c = finding(agent=AgentName.CODE, severity=Severity.LOW)  # exact repeat
    unique, removed = deduplicate([a, b, c])
    assert removed == 2 and len(unique) == 1
    assert unique[0].severity is Severity.HIGH  # highest severity kept
    assert unique[0].agent is AgentName.DOCUMENTATION
    assert unique[0].also_reported_by == [AgentName.CODE]


def test_dedup_keeps_distinct_lines_and_titles():
    items = [finding(line=1), finding(line=2), finding(title="Other")]
    assert deduplicate(items)[1] == 0


def test_sorting_is_by_severity_then_location(repo):
    results = [
        WorkerResult(
            agent=AgentName.CODE,
            status="ok",
            findings=[
                finding(severity=Severity.LOW, line=1, title="bbb"),
                finding(severity=Severity.HIGH, line=2, title="aaa"),
                finding(severity=Severity.LOW, line=1, title="aaa"),
            ],
        )
    ]
    findings, _, _ = finalize_findings(results, str(repo), Limits())
    assert [(f.severity.value, f.line, f.title) for f in findings] == [
        ("high", 2, "aaa"),
        ("low", 1, "aaa"),
        ("low", 1, "bbb"),
    ]


def test_failed_workers_contribute_nothing_but_do_not_hide_others(repo):
    results = [
        WorkerResult(agent=AgentName.CODE, status="ok", findings=[finding()], rejected_findings=2),
        WorkerResult(agent=AgentName.DEPENDENCIES, status="failed", error="boom"),
    ]
    findings, rejected, _ = finalize_findings(results, str(repo), Limits())
    assert len(findings) == 1 and rejected == 2


# -- Markdown -------------------------------------------------------------------------------


def summary(**kw) -> ReviewSummary:
    base = dict(
        repo="demo",
        mode=Mode.MOCK,
        routing=RoutingDecision(
            router="deterministic", source="rules", agents=(AgentName.CODE,), reason="all"
        ),
        workers=[WorkerResult(agent=AgentName.CODE, status="ok", model_calls=2, tool_calls=1)],
        findings=[finding(evidence="open('x')", verification="quote")],
    )
    return ReviewSummary(**{**base, **kw})


def test_markdown_contains_required_fields():
    md = render_markdown(summary())
    for needle in (
        "**Agent:** code",
        "`a.py:2`",
        "[MEDIUM]",
        "**Description:**",
        "**Recommendation:**",
        "quote verified",
        "no language model was used",
    ):
        assert needle in md


def test_markdown_is_deterministic():
    assert render_markdown(summary()) == render_markdown(summary())


def test_markdown_reports_failures_and_rejections():
    s = summary(
        workers=[WorkerResult(agent=AgentName.CODE, status="failed", error="Gateway | down")],
        findings=[],
        rejected_findings=3,
        duplicates_removed=1,
        skipped_agents={"dependencies": "no manifests"},
    )
    md = render_markdown(s)
    assert "FAILED: Gateway \\| down" in md and "No verified findings." in md
    assert "3 finding(s) were discarded" in md and "report is partial" in md
    assert "Skipped `dependencies`" in md


def test_markdown_states_that_verification_is_not_correctness():
    assert "does not prove" in render_markdown(summary())
