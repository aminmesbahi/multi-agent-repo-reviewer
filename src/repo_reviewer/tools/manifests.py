"""Manifest tools for the Dependency Analyst, plus a small requirement-line parser.

Manifests are only ever *read as text*. Nothing is installed, resolved or executed, and no
network lookup is made, so this module cannot make vulnerability claims.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from repo_reviewer.tools.filesystem import Sandbox, ToolError, read_file

_MANIFEST_RE = re.compile(r"requirements[^/]*\.txt")
_REQ_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(?P<spec>[^;#]*?)\s*(?:;.*)?(?:#.*)?$"
)


@dataclass(frozen=True)
class Requirement:
    name: str
    specifier: str

    @property
    def is_unpinned(self) -> bool:
        """No version constraint at all."""
        return not self.specifier

    @property
    def is_exact(self) -> bool:
        return bool(re.fullmatch(r"===?\s*[^,*\s]+", self.specifier))


def parse_requirement(text: str) -> Requirement | None:
    """Parse one PEP 508-style requirement string. Returns None for options and URLs."""
    stripped = text.strip()
    if not stripped or stripped.startswith(("#", "-", ".", "/")) or "://" in stripped:
        return None
    match = _REQ_RE.match(stripped)
    if match is None:
        return None
    return Requirement(match["name"], match["spec"].replace(" ", ""))


def is_manifest(rel_path: str) -> bool:
    name = PurePosixPath(rel_path).name.lower()
    return name == "pyproject.toml" or bool(_MANIFEST_RE.fullmatch(name))


def find_manifests(sandbox: Sandbox) -> list[str]:
    files, _ = sandbox.iter_files(".")
    return [path for path, _size in files if is_manifest(path)]


def list_manifests(sandbox: Sandbox) -> str:
    manifests = find_manifests(sandbox)
    if not manifests:
        return "No manifests (pyproject.toml, requirements*.txt) found."
    return "\n".join(manifests)


def read_manifest(sandbox: Sandbox, path: str) -> str:
    if not isinstance(path, str) or not is_manifest(path):
        raise ToolError("read_manifest only reads pyproject.toml and requirements*.txt")
    return read_file(sandbox, path)
