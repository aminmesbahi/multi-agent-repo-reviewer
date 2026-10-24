"""CLI behaviour: arguments, exit codes, safety prompts."""

from __future__ import annotations

import pytest
from conftest import SAMPLE, make_repo
from typer.testing import CliRunner

from repo_reviewer import cli
from repo_reviewer.cli import app, parse_agents
from repo_reviewer.gateway import GatewayError
from repo_reviewer.models import AgentName

runner = CliRunner()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("LITELLM_MASTER_KEY", "DEEPSEEK_API_KEY", "REVIEWER_GATEWAY_URL"):
        monkeypatch.delenv(key, raising=False)


def test_parse_agents_accepts_commas_and_repeats():
    assert parse_agents(["code,documentation", "dependencies"]) == [
        AgentName.CODE,
        AgentName.DOCUMENTATION,
        AgentName.DEPENDENCIES,
    ]
    assert parse_agents(None) is None and parse_agents([""]) is None


def test_unknown_agent_is_a_usage_error():
    result = runner.invoke(app, ["review", str(SAMPLE), "--agents", "security", "--mode", "mock"])
    assert (
        result.exit_code == 2 and "Valid agents: code, dependencies, documentation" in result.stderr
    )


def test_missing_path_exit_code():
    result = runner.invoke(app, ["review", "./definitely/not/here", "--mode", "mock"])
    assert result.exit_code == 4 and "is not a directory" in result.stderr


def test_local_mode_without_key_explains_how_to_fix():
    result = runner.invoke(app, ["review", str(SAMPLE), "--mode", "local"])
    assert result.exit_code == 2 and "LITELLM_MASTER_KEY" in result.stderr


def test_deepseek_warns_and_refuses_without_confirmation(monkeypatch):
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-test")
    result = runner.invoke(app, ["review", str(SAMPLE), "--mode", "deepseek"])  # stdin is not a TTY
    assert "external provider" in result.stderr
    assert result.exit_code == 2 and "--yes" in result.stderr


def test_deepseek_with_yes_proceeds_to_gateway(monkeypatch):
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-test")
    seen = {}

    class Down:
        def complete(self, messages, tools, *, timeout_s=None):
            seen["called"] = True
            raise GatewayError("cannot reach the LiteLLM Proxy")

    monkeypatch.setattr(cli, "build_client", lambda mode, yes: Down())
    result = runner.invoke(app, ["review", str(SAMPLE), "--mode", "deepseek", "--yes"])
    assert seen["called"]
    assert result.exit_code == 3  # every agent failed
    assert "FAILED" in result.stdout and "every agent failed" in result.stderr


def test_partial_failure_exit_code_1(monkeypatch):
    from repo_reviewer.testing.mock_model import MockModel

    class FlakyDeps(MockModel):
        def complete(self, messages, tools, *, timeout_s=None):
            if messages[0]["content"].startswith("AGENT: dependencies"):
                raise GatewayError("boom")
            return super().complete(messages, tools, timeout_s=timeout_s)

    monkeypatch.setattr(cli, "build_client", lambda mode, yes: FlakyDeps())
    result = runner.invoke(app, ["review", str(SAMPLE), "--mode", "mock"])
    assert result.exit_code == 1
    assert "report is partial" in result.stdout


def test_no_applicable_agent_is_usage_error(tmp_path):
    repo = make_repo(tmp_path / "r", {"README.md": "# x\n"})
    result = runner.invoke(app, ["review", str(repo), "--agents", "code", "--mode", "mock"])
    assert result.exit_code == 2 and "no Python files" in result.stderr


def test_verbose_enables_debug_logging():
    quiet = runner.invoke(app, ["review", str(SAMPLE), "--mode", "mock"])
    loud = runner.invoke(app, ["review", str(SAMPLE), "--mode", "mock", "--verbose"])
    assert "DEBUG" not in quiet.stderr and "DEBUG" in loud.stderr


def test_logs_do_not_contain_file_contents():
    loud = runner.invoke(app, ["review", str(SAMPLE), "--mode", "mock", "--verbose"])
    assert "json.load" not in loud.stderr and "pydantic==2.9.2" not in loud.stderr


def test_smoke_refuses_mock_mode():
    result = runner.invoke(app, ["smoke", "--mode", "mock"])
    assert result.exit_code == 2


def test_help_lists_documented_options():
    result = runner.invoke(app, ["review", "--help"])
    for flag in ("--agents", "--mode", "--output", "--json", "--verbose", "--router", "--request"):
        assert flag in result.stdout


# -- routing options ------------------------------------------------------------------------


def test_decision_router_requires_request():
    result = runner.invoke(app, ["review", str(SAMPLE), "--mode", "mock", "--router", "decision"])
    assert result.exit_code == 2 and "--request" in result.stderr


def test_decision_router_conflicts_with_explicit_agents():
    result = runner.invoke(
        app,
        [
            "review",
            str(SAMPLE),
            "--mode",
            "mock",
            "--router",
            "decision",
            "--request",
            "x",
            "--agents",
            "code",
        ],
    )
    assert result.exit_code == 2 and "mutually exclusive" in result.stderr


def test_decision_router_end_to_end_with_mock_backend():
    result = runner.invoke(
        app,
        [
            "review",
            str(SAMPLE),
            "--mode",
            "mock",
            "--router",
            "decision",
            "--json",
            "--request",
            "check the README installation steps and the pinned dependencies",
        ],
    )
    assert result.exit_code == 0, result.stderr
    import json

    data = json.loads(result.stdout)
    assert sorted(w["agent"] for w in data["workers"]) == ["dependencies", "documentation"]
    assert data["routing"]["source"] == "decision_model"
    assert data["skipped_agents"]["code"].startswith("not selected")


def test_default_router_ignores_request_free_runs():
    result = runner.invoke(app, ["review", str(SAMPLE), "--mode", "mock", "--json"])
    import json

    assert json.loads(result.stdout)["routing"]["router"] == "deterministic"


def test_systemone_backend_refuses_remote_host_by_default(monkeypatch):
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-test")
    monkeypatch.setenv("DECISION_API_BASE", "https://api.example.com")
    result = runner.invoke(
        app,
        [
            "review",
            str(SAMPLE),
            "--mode",
            "mock",
            "--router",
            "decision",
            "--decision-backend",
            "systemone",
            "--request",
            "readme",
        ],
    )
    assert result.exit_code == 2 and "not local" in result.stderr
