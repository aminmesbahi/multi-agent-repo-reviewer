"""Tool registry and the per-agent permission boundary.

An agent only ever receives the OpenAI tool schema for tools in its allowlist, and
``ToolBox.call`` re-checks the allowlist, so a model that hallucinates a tool name
(or one it was never offered) gets an error instead of access.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from repo_reviewer.tools import filesystem as fs
from repo_reviewer.tools import manifests as mf
from repo_reviewer.tools.filesystem import Sandbox, ToolError


class ToolPermissionError(ToolError):
    """The agent asked for a tool that is not on its allowlist."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., str]

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _obj(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


_PATH = {"type": "string", "description": "Path relative to the repository root."}

ALL_TOOLS: dict[str, Tool] = {
    tool.name: tool
    for tool in (
        Tool(
            "list_files",
            "List files in the repository (sorted, excludes secrets and VCS data).",
            _obj(
                {
                    "subdir": {"type": "string", "description": "Directory to list. Default '.'"},
                    "suffix": {"type": "string", "description": "Only files ending with this."},
                }
            ),
            fs.list_files,
        ),
        Tool(
            "read_file",
            "Read a text file with line numbers. Use start_line/end_line for large files.",
            _obj(
                {
                    "path": _PATH,
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                ["path"],
            ),
            fs.read_file,
        ),
        Tool(
            "list_manifests",
            "List dependency manifests: pyproject.toml and requirements*.txt.",
            _obj({}),
            mf.list_manifests,
        ),
        Tool(
            "read_manifest",
            "Read one dependency manifest with line numbers.",
            _obj({"path": _PATH}, ["path"]),
            mf.read_manifest,
        ),
        Tool(
            "read_readme",
            "Read README.md with line numbers.",
            _obj({}),
            fs.read_readme,
        ),
        Tool(
            "inspect_project_metadata",
            "Return parsed pyproject.toml metadata and basic project facts as JSON.",
            _obj({}),
            fs.inspect_project_metadata,
        ),
    )
}


class ToolBox:
    """The tools one agent may use, bound to one sandbox."""

    def __init__(self, sandbox: Sandbox, allowed: frozenset[str]) -> None:
        unknown = allowed - ALL_TOOLS.keys()
        if unknown:
            raise ValueError(f"unknown tools in allowlist: {sorted(unknown)}")
        self.sandbox = sandbox
        self.allowed = allowed

    def openai_tools(self) -> list[dict[str, Any]]:
        return [ALL_TOOLS[name].openai_schema() for name in sorted(self.allowed)]

    def call(self, name: str, raw_arguments: str | dict[str, Any] | None) -> str:
        """Execute a tool call from the model. Raises ToolError for anything refused."""
        if name not in self.allowed:
            raise ToolPermissionError(f"tool '{name}' is not available to this agent")
        tool = ALL_TOOLS[name]
        arguments = _parse_arguments(raw_arguments)
        allowed_args = tool.parameters["properties"].keys()
        extra = set(arguments) - allowed_args
        if extra:
            raise ToolError(f"unexpected argument(s) for {name}: {sorted(extra)}")
        try:
            return tool.handler(self.sandbox, **arguments)
        except TypeError as exc:  # missing/ill-typed argument supplied by the model
            raise ToolError(f"invalid arguments for {name}") from exc


def _parse_arguments(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolError("tool arguments are not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ToolError("tool arguments must be a JSON object")
    return parsed
