"""MOCK mode: a deterministic stand-in for the chat model. This is NOT an LLM.

``MockModel`` implements the same ``ChatClient`` protocol as the LiteLLM gateway, so the
real agent loop, tool sandbox, graph and report pipeline all run unchanged. It makes
scripted tool-call decisions (list, then read, then answer) and derives its findings from the
tool output it actually receives, using plain AST/regex checks. It is useful for CI and demos;
it says nothing about how a real model would behave.
"""

from __future__ import annotations

import ast
import json
import re
import tomllib
from typing import Any

from repo_reviewer.gateway import ModelReply, ToolCall
from repo_reviewer.tools.manifests import parse_requirement

_ROW_RE = re.compile(r"^\s*(\d+)\| ?(.*)$")
_HEADER_RE = re.compile(r"^# (.+?) \(lines ")
MAX_FILES_TO_READ = 6


def parse_numbered(content: str) -> tuple[str | None, dict[int, str]]:
    """Inverse of ``format_numbered``: (path, {line number: text})."""
    path = None
    rows: dict[int, str] = {}
    for line in content.splitlines():
        header = _HEADER_RE.match(line)
        if header and path is None:
            path = header.group(1)
            continue
        row = _ROW_RE.match(line)
        if row:
            rows[int(row.group(1))] = row.group(2)
    return path, rows


class MockModel:
    """Scripted ChatClient. Stateless: the next action is a function of the transcript."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        timeout_s: float | None = None,
    ) -> ModelReply:
        agent = messages[0]["content"].split("\n", 1)[0].removeprefix("AGENT: ").strip()
        tool_results = [m["content"] for m in messages if m["role"] == "tool"]
        offered = {t["function"]["name"] for t in tools or []}
        script = {
            "code": self._code,
            "dependencies": self._dependencies,
            "documentation": self._documentation,
        }.get(agent)
        if script is None:
            return ModelReply(content=json.dumps({"findings": []}))
        return script(tool_results, offered)

    # -- Code Reviewer -----------------------------------------------------------------

    def _code(self, results: list[str], offered: set[str]) -> ModelReply:
        if not results and "list_files" in offered:
            return _call("call_list", "list_files", {"suffix": ".py"})
        if len(results) == 1 and "read_file" in offered:
            paths = [
                line.split(" (", 1)[0]
                for line in results[0].splitlines()
                if line.endswith(" bytes)") and not line.startswith("  ")
            ]
            sources = [p for p in paths if "test" not in p.lower().split("/")[0]]
            return ModelReply(
                content=None,
                tool_calls=[
                    ToolCall(f"call_read_{i}", "read_file", json.dumps({"path": p}))
                    for i, p in enumerate(sources[:MAX_FILES_TO_READ])
                ],
            )
        findings: list[dict[str, Any]] = []
        for result in results[1:]:
            path, rows = parse_numbered(result)
            if path:
                findings += _review_python(path, rows)
        return _final(findings)

    # -- Dependency Analyst ------------------------------------------------------------

    def _dependencies(self, results: list[str], offered: set[str]) -> ModelReply:
        if not results and "list_manifests" in offered:
            return _call("call_list", "list_manifests", {})
        if len(results) == 1 and "read_manifest" in offered:
            paths = [p.strip() for p in results[0].splitlines() if p.strip() and " " not in p]
            return ModelReply(
                content=None,
                tool_calls=[
                    ToolCall(f"call_read_{i}", "read_manifest", json.dumps({"path": p}))
                    for i, p in enumerate(paths)
                ],
            )
        findings: list[dict[str, Any]] = []
        pins: dict[str, tuple[str, str, int]] = {}
        for result in results[1:]:
            path, rows = parse_numbered(result)
            if path:
                findings += _review_manifest(path, rows, pins)
        return _final(findings)

    # -- Documentation Reviewer --------------------------------------------------------

    def _documentation(self, results: list[str], offered: set[str]) -> ModelReply:
        if not results and {"read_readme", "inspect_project_metadata"} <= offered:
            return ModelReply(
                content=None,
                tool_calls=[
                    ToolCall("call_readme", "read_readme", "{}"),
                    ToolCall("call_meta", "inspect_project_metadata", "{}"),
                ],
            )
        readme, metadata = (results + ["", ""])[:2] if results else ("", "")
        return _final(_review_readme(readme, metadata))


# -- helpers: reply construction -----------------------------------------------------------


def _call(call_id: str, name: str, arguments: dict[str, Any]) -> ModelReply:
    return ModelReply(content=None, tool_calls=[ToolCall(call_id, name, json.dumps(arguments))])


def _final(findings: list[dict[str, Any]]) -> ModelReply:
    return ModelReply(content=json.dumps({"findings": findings}))


def _finding(
    file: str,
    line: int | None,
    severity: str,
    title: str,
    description: str,
    recommendation: str,
    evidence: str | None,
) -> dict[str, Any]:
    return {
        "file": file,
        "line": line,
        "severity": severity,
        "title": title,
        "description": description,
        "recommendation": recommendation,
        "evidence": evidence,
    }


# -- helpers: the "analysis" (plain static checks over tool output) ---------------------------

_IO_METHODS = {"read_text", "read_bytes", "write_text", "write_bytes"}


class _IOVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.try_depth = 0
        self.unguarded_io: list[ast.Call] = []
        self.bare_excepts: list[ast.ExceptHandler] = []
        self.mutable_defaults: list[int] = []

    def visit_Try(self, node: ast.Try) -> None:  # noqa: N802
        self.try_depth += 1
        for child in node.body:
            self.visit(child)
        self.try_depth -= 1
        for handler in node.handlers:
            if handler.type is None:
                self.bare_excepts.append(handler)
            self.visit(handler)
        for child in (*node.orelse, *node.finalbody):
            self.visit(child)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func = node.func
        is_open = isinstance(func, ast.Name) and func.id == "open"
        is_path_io = isinstance(func, ast.Attribute) and func.attr in _IO_METHODS
        if (is_open or is_path_io) and self.try_depth == 0:
            self.unguarded_io.append(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        defaults = [*node.args.defaults, *[d for d in node.args.kw_defaults if d]]
        for default in defaults:
            if isinstance(default, ast.List | ast.Dict | ast.Set):
                self.mutable_defaults.append(default.lineno)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815


def _review_python(path: str, rows: dict[int, str]) -> list[dict[str, Any]]:
    if not rows:
        return []
    source = "\n".join(rows.get(n, "") for n in range(1, max(rows) + 1))
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    visitor = _IOVisitor()
    visitor.visit(tree)
    out: list[dict[str, Any]] = []
    for call in visitor.unguarded_io:
        text = rows.get(call.lineno, "").strip()
        out.append(
            _finding(
                path,
                call.lineno,
                "medium",
                "File I/O without exception handling",
                "This call can raise OSError (missing file, permissions, disk errors) and "
                "nothing in the enclosing code handles it, so a bad path crashes the caller.",
                "Wrap the I/O in try/except OSError (and ValueError for parsing) and return "
                "a clear error or default.",
                text,
            )
        )
    for handler in visitor.bare_excepts:
        out.append(
            _finding(
                path,
                handler.lineno,
                "medium",
                "Bare except clause",
                "A bare 'except:' also catches KeyboardInterrupt and SystemExit and hides bugs.",
                "Catch specific exception types.",
                rows.get(handler.lineno, "").strip(),
            )
        )
    for lineno in visitor.mutable_defaults:
        out.append(
            _finding(
                path,
                lineno,
                "low",
                "Mutable default argument",
                "A mutable default value is shared between calls.",
                "Use None as the default and create the container inside the function.",
                rows.get(lineno, "").strip(),
            )
        )
    return out


def _review_manifest(
    path: str, rows: dict[int, str], pins: dict[str, tuple[str, str, int]]
) -> list[dict[str, Any]]:
    entries: list[tuple[int, str, str]] = []  # (line, evidence text, requirement string)
    if path.lower().endswith(".toml"):
        try:
            data = tomllib.loads(
                "\n".join(rows.get(n, "") for n in range(1, max(rows, default=0) + 1))
            )
        except tomllib.TOMLDecodeError:
            return []
        project = data.get("project", {})
        specs = [*project.get("dependencies", [])]
        for group in project.get("optional-dependencies", {}).values():
            specs += group
        for spec in specs:
            line = next((n for n, text in sorted(rows.items()) if f'"{spec}"' in text), None)
            if line:
                entries.append((line, spec, spec))
    else:
        entries = [(n, text.strip(), text) for n, text in sorted(rows.items())]

    out: list[dict[str, Any]] = []
    for line, evidence, spec in entries:
        req = parse_requirement(spec)
        if req is None:
            continue
        evidence = evidence.split("#", 1)[0].strip()
        if req.is_unpinned:
            out.append(
                _finding(
                    path,
                    line,
                    "medium",
                    f"Unpinned dependency '{req.name}'",
                    f"'{req.name}' has no version constraint, so any release can be installed "
                    "and builds are not reproducible.",
                    f"Pin it, e.g. '{req.name}==<tested version>', and keep a lockfile.",
                    evidence,
                )
            )
        elif req.is_exact:
            key = req.name.lower().replace("_", "-")
            previous = pins.get(key)
            if previous and previous[1] != req.specifier:
                out.append(
                    _finding(
                        path,
                        line,
                        "medium",
                        f"Conflicting pins for '{req.name}'",
                        f"'{req.name}' is pinned to {req.specifier} here but to {previous[1]} "
                        f"in {previous[0]}:{previous[2]}.",
                        "Use a single version across manifests.",
                        evidence,
                    )
                )
            pins.setdefault(key, (path, req.specifier, line))
    return out


def _review_readme(readme: str, metadata: str) -> list[dict[str, Any]]:
    path, rows = parse_numbered(readme)
    if not path:
        return []  # README missing: nothing to anchor a finding to in this scripted check
    try:
        meta = json.loads(metadata).get("pyproject") or {}
    except json.JSONDecodeError:
        meta = {}
    text = "\n".join(rows.get(n, "") for n in sorted(rows))
    lowered = text.lower()
    anchor_line = next((n for n in sorted(rows) if rows[n].strip()), None)
    anchor = rows[anchor_line].strip() if anchor_line else None
    out: list[dict[str, Any]] = []

    def add(severity: str, title: str, description: str, recommendation: str) -> None:
        out.append(
            _finding(path, anchor_line, severity, title, description, recommendation, anchor)
        )

    if not re.search(r"^#+\s.*install", text, re.IGNORECASE | re.MULTILINE) and not re.search(
        r"pip install|uv (sync|pip|add)|poetry install", lowered
    ):
        add(
            "medium",
            "README has no installation instructions",
            "A reader cannot tell how to install the project.",
            "Add an Installation section with the exact install command.",
        )
    requires = meta.get("requires_python")
    version = re.search(r"\d+(\.\d+)+", requires or "")
    if version and version.group(0) not in text:
        add(
            "low",
            "README does not state the required Python version",
            f"pyproject.toml requires Python {requires}, but the README never mentions it.",
            f"State the prerequisite 'Python {requires}' near the install steps.",
        )
    scripts = meta.get("console_scripts") or []
    if scripts and not any(s in text for s in scripts):
        add(
            "medium",
            "README does not document the command-line entry point",
            f"pyproject.toml defines the script(s) {', '.join(scripts)}, which the README omits.",
            "Add a Usage section showing how to run the command.",
        )
    elif not re.search(
        r"^#+\s.*(usage|quick ?start|getting started|example)", text, re.IGNORECASE | re.MULTILINE
    ):
        add(
            "medium",
            "README has no usage section",
            "A reader cannot tell how to use the project after installing it.",
            "Add a Usage or Quickstart section with a runnable example.",
        )
    return out
