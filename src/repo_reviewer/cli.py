"""Command-line interface. Progress and diagnostics go to stderr; the report goes to stdout
(or ``--output``). Exit codes:

    0  review completed, every selected agent succeeded (findings do not change the code)
    1  review completed but at least one agent failed (partial report)
    2  usage or configuration error
    3  every selected agent failed (e.g. gateway down)
    4  the target path is not a readable directory
"""

from __future__ import annotations

import logging
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from repo_reviewer.config import ConfigError, decision_settings, gateway_settings
from repo_reviewer.gateway import ChatClient, GatewayError, LiteLLMGateway
from repo_reviewer.models import AgentName, Limits, Mode
from repo_reviewer.orchestrator.graph import run_review
from repo_reviewer.orchestrator.planner import NoApplicableAgents
from repo_reviewer.routing.backends import MockDecisionBackend, SystemOneBackend
from repo_reviewer.routing.base import AgentRouter
from repo_reviewer.routing.decision import DEFAULT_SELECT_THRESHOLD, DecisionModelRouter
from repo_reviewer.routing.rules import RuleBasedRouter
from repo_reviewer.testing.mock_model import MockModel
from repo_reviewer.tools.filesystem import ToolError

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Read-only multi-agent Python repository reviewer.",
)
err = typer.echo


class RouterKind(StrEnum):
    DETERMINISTIC = "deterministic"
    DECISION = "decision"


class DecisionBackendKind(StrEnum):
    MOCK = "mock"
    SYSTEMONE = "systemone"


class Fail(Exception):
    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


def setup_logging(verbose: bool) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for noisy in ("httpx", "httpx2", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def parse_agents(values: list[str] | None) -> list[AgentName] | None:
    if not values:
        return None
    names: list[AgentName] = []
    for chunk in values:
        for raw in chunk.split(","):
            raw = raw.strip().lower()
            if not raw:
                continue
            try:
                names.append(AgentName(raw))
            except ValueError:
                valid = ", ".join(a.value for a in AgentName)
                raise Fail(f"unknown agent '{raw}'. Valid agents: {valid}", 2) from None
    return names or None


def build_client(mode: Mode, assume_yes: bool) -> ChatClient:
    if mode is Mode.MOCK:
        return MockModel()
    settings = gateway_settings(mode)
    if mode is Mode.DEEPSEEK:
        err(
            "WARNING: DEEPSEEK mode sends excerpts of the reviewed source code to the DeepSeek "
            "API (an external provider) through your LiteLLM Proxy.",
            err=True,
        )
        if not assume_yes:
            if not sys.stdin.isatty():
                raise Fail("refusing to send code externally without confirmation; pass --yes", 2)
            if not typer.confirm("Continue?", err=True):
                raise Fail("aborted by user", 2)
    err(f"model alias '{settings.model}' via {settings.base_url}", err=True)
    return LiteLLMGateway(settings)


def build_router(
    kind: RouterKind,
    backend_kind: DecisionBackendKind | None,
    mode: Mode,
    threshold: float,
) -> AgentRouter:
    if kind is RouterKind.DETERMINISTIC:
        return RuleBasedRouter()
    backend_kind = backend_kind or (
        DecisionBackendKind.MOCK if mode is Mode.MOCK else DecisionBackendKind.SYSTEMONE
    )
    if backend_kind is DecisionBackendKind.MOCK:
        err("decision router: MOCK backend (lexical scorer, not a model)", err=True)
        return DecisionModelRouter(MockDecisionBackend(), threshold)
    settings = decision_settings()
    err(
        f"decision router: EXPERIMENTAL systemone backend at {settings.api_base} "
        f"(model '{settings.model}'). Only the request text and coarse counts are sent.",
        err=True,
    )
    return DecisionModelRouter(SystemOneBackend(settings), threshold)


@app.command()
def review(
    path: Annotated[Path, typer.Argument(help="Repository directory to review (read-only).")],
    mode: Annotated[Mode, typer.Option(help="mock | local | deepseek")] = Mode.MOCK,
    agents: Annotated[
        list[str] | None,
        typer.Option(help="Agents to run: code, dependencies, documentation (comma-separated)."),
    ] = None,
    output: Annotated[Path | None, typer.Option(help="Write the report here, not stdout.")] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit JSON instead of Markdown.")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", help="Debug logging on stderr.")] = False,
    router: Annotated[
        RouterKind, typer.Option(help="How agents are chosen.")
    ] = RouterKind.DETERMINISTIC,
    request: Annotated[
        str | None, typer.Option(help="Free-text request; used by the router, never by agents.")
    ] = None,
    decision_backend: Annotated[
        DecisionBackendKind | None,
        typer.Option(help="Backend for --router decision (default: mock in MOCK mode)."),
    ] = None,
    threshold: Annotated[
        float, typer.Option(help="Decision router score threshold (illustrative default).")
    ] = DEFAULT_SELECT_THRESHOLD,
    max_model_calls: Annotated[int, typer.Option(min=1, max=50)] = 8,
    max_tool_calls: Annotated[int, typer.Option(min=0, max=100)] = 12,
    timeout: Annotated[float, typer.Option(help="Total review time budget in seconds.")] = 300.0,
    yes: Annotated[bool, typer.Option("--yes", help="Skip the DEEPSEEK confirmation.")] = False,
) -> None:
    """Review PATH with the selected agents and print a Markdown (or JSON) report."""
    setup_logging(verbose)
    try:
        code = _review(
            path,
            mode,
            parse_agents(agents),
            output,
            json_output,
            router,
            request,
            decision_backend,
            threshold,
            yes,
            Limits(
                max_model_calls=max_model_calls,
                max_tool_calls=max_tool_calls,
                total_timeout_s=timeout,
            ),
        )
    except Fail as exc:
        err(f"error: {exc}", err=True)
        code = exc.code
    except ConfigError as exc:
        err(f"configuration error: {exc}", err=True)
        code = 2
    raise typer.Exit(code)


def _review(
    path: Path,
    mode: Mode,
    requested: list[AgentName] | None,
    output: Path | None,
    as_json: bool,
    router_kind: RouterKind,
    request: str | None,
    backend_kind: DecisionBackendKind | None,
    threshold: float,
    assume_yes: bool,
    limits: Limits,
) -> int:
    if not path.is_dir():
        raise Fail(f"'{path}' is not a directory", 4)
    if router_kind is RouterKind.DECISION and not request:
        raise Fail('--router decision needs --request "<what you want reviewed>"', 2)
    if router_kind is RouterKind.DECISION and requested:
        raise Fail("--agents and --router decision are mutually exclusive", 2)
    if not 0.0 < threshold <= 1.0:
        raise Fail("--threshold must be in (0, 1]", 2)

    client = build_client(mode, assume_yes)
    router = build_router(router_kind, backend_kind, mode, threshold)
    err(f"reviewing {path} in {mode.value.upper()} mode", err=True)
    try:
        result = run_review(
            str(path),
            mode=mode,
            client=client,
            router=router,
            requested_agents=requested,
            request_text=request,
            limits=limits,
        )
    except NoApplicableAgents as exc:
        raise Fail(str(exc), 2) from exc
    except ToolError as exc:
        raise Fail(str(exc), 4) from exc

    text = result.summary.model_dump_json(indent=2) if as_json else result.report
    if output:
        output.write_text(text, encoding="utf-8")
        err(f"report written to {output}", err=True)
    else:
        _write_stdout(text)

    failed = result.summary.failed_workers
    if failed and len(failed) == len(result.summary.workers):
        err("error: every agent failed; see the report for reasons", err=True)
        return 3
    return 1 if failed else 0


def _write_stdout(text: str) -> None:
    stream = sys.stdout
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")  # Windows consoles default to a legacy codepage
    stream.write(text if text.endswith("\n") else text + "\n")


@app.command()
def smoke(
    mode: Annotated[Mode, typer.Option(help="local | deepseek")] = Mode.LOCAL,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    """Optional live check: is the gateway up, is the alias served, does the model call tools?

    Sends a synthetic prompt only; no repository content is involved.
    """
    setup_logging(verbose)
    if mode is Mode.MOCK:
        err("error: smoke tests real inference; use --mode local or --mode deepseek", err=True)
        raise typer.Exit(2)
    try:
        settings = gateway_settings(mode)
        gateway = LiteLLMGateway(settings)
        models = gateway.list_models()
        err(f"gateway models: {', '.join(models) or '(none)'}", err=True)
        if settings.model not in models:
            raise Fail(f"alias '{settings.model}' is not served by the proxy", 3)
        probe = [
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ]
        reply = gateway.complete(
            [{"role": "user", "content": "Call the list_files tool now. Do not answer in text."}],
            probe,
        )
    except ConfigError as exc:
        err(f"configuration error: {exc}", err=True)
        raise typer.Exit(2) from exc
    except Fail as exc:
        err(f"error: {exc}", err=True)
        raise typer.Exit(exc.code) from exc
    except GatewayError as exc:
        err(f"error: {exc}", err=True)
        raise typer.Exit(3) from exc
    if reply.tool_calls:
        typer.echo(f"OK: '{settings.model}' returned a tool call ({reply.tool_calls[0].name}).")
        raise typer.Exit(0)
    typer.echo(
        f"FAIL: '{settings.model}' answered in text instead of calling a tool. It may not "
        "support tool calling reliably; see README 'Tool-calling limitations'."
    )
    raise typer.Exit(3)
