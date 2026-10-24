"""Typed data shared by the graph, the agents and the report.

Everything a model produces is parsed into these Pydantic models before it can
influence the report. Schema validity is *not* factual correctness: see
``orchestrator/validation.py`` for the checks against the real files.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Mode(StrEnum):
    MOCK = "mock"
    LOCAL = "local"
    DEEPSEEK = "deepseek"


class AgentName(StrEnum):
    """The explicit allowlist of agent identifiers. Nothing else may be routed to."""

    CODE = "code"
    DEPENDENCIES = "dependencies"
    DOCUMENTATION = "documentation"


class Severity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def rank(self) -> int:
        return ("info", "low", "medium", "high").index(self.value)


class Limits(BaseModel):
    """Hard bounds applied to every worker. All are configurable via the CLI/env."""

    model_config = ConfigDict(frozen=True)

    max_model_calls: int = Field(8, ge=1, le=50)
    max_tool_calls: int = Field(12, ge=0, le=100)
    max_file_bytes: int = Field(200_000, ge=1_000)
    max_files: int = Field(60, ge=1, le=1_000)
    max_tool_output_chars: int = Field(12_000, ge=500)
    max_context_chars: int = Field(60_000, ge=2_000)
    max_findings: int = Field(20, ge=1, le=200)
    total_timeout_s: float = Field(300.0, gt=0)


class RawFinding(BaseModel):
    """What a model is allowed to say. The agent name is added by the harness."""

    model_config = ConfigDict(extra="ignore")

    file: str = Field(min_length=1, max_length=300)
    line: int | None = Field(default=None, ge=1)
    severity: Severity
    title: str = Field(min_length=3, max_length=160)
    description: str = Field(min_length=3, max_length=1_500)
    recommendation: str = Field(min_length=3, max_length=1_000)
    evidence: str | None = Field(default=None, max_length=500)

    @field_validator("file")
    @classmethod
    def _normalise_file(cls, value: str) -> str:
        return value.strip().replace("\\", "/").removeprefix("./")


class Finding(RawFinding):
    agent: AgentName
    # quote: evidence text found near the cited line; line: line exists; file: file exists
    verification: Literal["quote", "line", "file"] = "file"
    also_reported_by: list[AgentName] = Field(default_factory=list)


class WorkerInput(BaseModel):
    """The only data a worker receives through ``Send`` (isolated local context)."""

    agent: AgentName
    repo_root: str
    limits: Limits
    deadline: float = Field(description="Absolute time.monotonic() deadline for this review.")


class WorkerResult(BaseModel):
    agent: AgentName
    status: Literal["ok", "failed"]
    findings: list[Finding] = Field(default_factory=list)
    rejected_findings: int = 0
    error: str | None = None
    model_calls: int = 0
    tool_calls: int = 0
    duration_s: float = 0.0


class RoutingDecision(BaseModel):
    """Which agents to run and why. Produced by an ``AgentRouter`` (see ``routing/``)."""

    router: str
    source: Literal["cli", "rules", "decision_model", "fallback"]
    agents: tuple[AgentName, ...]
    reason: str
    scores: dict[AgentName, float] | None = None
    fallback_reason: str | None = None
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None

    @field_validator("agents")
    @classmethod
    def _non_empty_unique(cls, value: tuple[AgentName, ...]) -> tuple[AgentName, ...]:
        if not value:
            raise ValueError("a routing decision must select at least one agent")
        if len(set(value)) != len(value):
            raise ValueError("duplicate agent in routing decision")
        return value


class ReviewSummary(BaseModel):
    repo: str
    mode: Mode
    routing: RoutingDecision
    skipped_agents: dict[str, str] = Field(default_factory=dict)
    workers: list[WorkerResult]
    findings: list[Finding]
    rejected_findings: int = 0
    duplicates_removed: int = 0

    @property
    def failed_workers(self) -> list[WorkerResult]:
        return [w for w in self.workers if w.status == "failed"]
