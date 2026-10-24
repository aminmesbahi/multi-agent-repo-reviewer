from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from repo_reviewer.gateway import ModelReply
from repo_reviewer.models import AgentName, Limits, WorkerInput

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample_project"


@pytest.fixture
def sample() -> Path:
    return SAMPLE


@pytest.fixture
def limits() -> Limits:
    return Limits()


def make_repo(root: Path, files: dict[str, str]) -> Path:
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return root


def worker_input(agent: AgentName, repo: Path, limits: Limits | None = None) -> WorkerInput:
    return WorkerInput(
        agent=agent,
        repo_root=str(repo),
        limits=limits or Limits(),
        deadline=time.monotonic() + 60,
    )


class ScriptedClient:
    """A ChatClient that replays prepared replies and records what it was asked."""

    def __init__(self, replies: list[ModelReply]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages, tools, *, timeout_s=None) -> ModelReply:
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        if not self.replies:
            raise AssertionError("ScriptedClient ran out of replies")
        return self.replies.pop(0)
