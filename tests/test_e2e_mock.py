"""End-to-end MOCK mode: real CLI, real graph, real sandbox, no network, no keys."""

from __future__ import annotations

import json
import socket

import pytest
from conftest import SAMPLE
from typer.testing import CliRunner

from repo_reviewer.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("network access attempted in MOCK mode")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    for key in ("LITELLM_MASTER_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(key, raising=False)


def test_mock_review_finds_every_planted_defect():
    result = runner.invoke(app, ["review", str(SAMPLE), "--mode", "mock", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    manifest = json.loads((SAMPLE / "expected_findings.json").read_text(encoding="utf-8"))
    for expected in manifest["expected"]:
        hits = [
            f
            for f in data["findings"]
            if f["agent"] == expected["agent"]
            and f["file"] == expected["file"]
            and expected["title_contains"].lower() in f["title"].lower()
            and expected.get("line", f["line"]) == f["line"]
        ]
        assert hits, f"planted defect not found: {expected['id']}"
        assert hits[0]["verification"] in {"quote", "line"}


def test_mock_review_markdown_report_to_stdout():
    result = runner.invoke(app, ["review", str(SAMPLE), "--mode", "mock"])
    assert result.exit_code == 0
    assert result.stdout.startswith("# Repository review")
    assert "Unpinned dependency 'requests'" in result.stdout
    assert "File I/O without exception handling" in result.stdout
    assert "no installation instructions" in result.stdout


def test_progress_goes_to_stderr_and_report_to_stdout():
    result = runner.invoke(app, ["review", str(SAMPLE), "--mode", "mock"])
    assert "reviewing" in result.stderr and "reviewing" not in result.stdout
    assert "# Repository review" not in result.stderr


def test_single_agent_selection():
    result = runner.invoke(
        app, ["review", str(SAMPLE), "--agents", "documentation", "--mode", "mock", "--json"]
    )
    data = json.loads(result.stdout)
    assert [w["agent"] for w in data["workers"]] == ["documentation"]
    assert {f["agent"] for f in data["findings"]} == {"documentation"}


def test_output_file(tmp_path):
    out = tmp_path / "report.md"
    result = runner.invoke(app, ["review", str(SAMPLE), "--mode", "mock", "--output", str(out)])
    assert result.exit_code == 0 and result.stdout == ""
    assert out.read_text(encoding="utf-8").startswith("# Repository review")


def test_mock_output_is_reproducible():
    args = ["review", str(SAMPLE), "--mode", "mock"]
    assert runner.invoke(app, args).stdout == runner.invoke(app, args).stdout


def test_review_does_not_modify_the_reviewed_repository():
    def snapshot():
        return {
            p: p.read_bytes()
            for p in SAMPLE.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        }

    before = snapshot()
    runner.invoke(app, ["review", str(SAMPLE), "--mode", "mock"])
    assert snapshot() == before
