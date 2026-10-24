"""Read-only, sandboxed filesystem access for agents.

The repository under review is untrusted input. Every path an agent supplies goes
through ``Sandbox.resolve``; every read goes through ``Sandbox.read_lines``.
Nothing here can write, execute or follow a symlink.

What this protects against: path traversal, absolute paths, symlink/junction escapes,
secret-looking files, binaries, huge files and unbounded reads.
What it does NOT protect against: a hostile process racing on the same directory while a
review runs (we open with O_NOFOLLOW where the OS has it and re-check with fstat, which
narrows but does not eliminate that window), or secrets stored in files with ordinary names.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import stat
import tomllib
from pathlib import Path, PurePosixPath

from repo_reviewer.models import Limits

EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".tox",
        ".nox",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "dist",
        "build",
        ".ssh",
        ".aws",
        ".gnupg",
        ".idea",
        ".vscode",
    }
)
SECRET_NAME_PATTERNS = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*.kdbx",
    "id_rsa*",
    "id_dsa*",
    "id_ecdsa*",
    "id_ed25519*",
    ".netrc",
    ".pypirc",
    ".npmrc",
    "credentials*",
    "secrets*",
    "*.secret",
    "*.secrets",
)
MAX_WALK_DEPTH = 8
MAX_LINE_CHARS = 500
_DRIVE_RE = re.compile(r"^[A-Za-z]:")


class ToolError(Exception):
    """A tool refused or failed. The message is safe to show to the model."""


class PathRejected(ToolError):
    """The requested path is outside the sandbox or excluded by policy."""


class LimitExceeded(ToolError):
    """A configured bound (file count, size) was hit."""


def is_secret_name(name: str) -> bool:
    lowered = name.lower()
    return any(fnmatch.fnmatchcase(lowered, pattern) for pattern in SECRET_NAME_PATTERNS)


def _is_link(path: Path) -> bool:
    return path.is_symlink() or os.path.isjunction(path)


class Sandbox:
    """A read-only view of one repository root with per-instance read accounting."""

    def __init__(self, root: str | os.PathLike[str], limits: Limits) -> None:
        try:
            resolved = Path(root).resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise ToolError(f"repository root does not exist: {root}") from exc
        if not resolved.is_dir():
            raise ToolError(f"repository root is not a directory: {root}")
        self.root = resolved
        self.limits = limits
        self._files_read: set[str] = set()

    # -- path handling -------------------------------------------------------------

    def resolve(self, rel: object) -> tuple[Path, str]:
        """Validate an agent-supplied relative path. Returns (absolute path, posix rel path)."""
        if not isinstance(rel, str) or not rel or len(rel) > 512 or "\x00" in rel:
            raise PathRejected("path must be a non-empty string without NUL bytes (max 512 chars)")
        normalised = rel.replace("\\", "/")
        if normalised.startswith("/") or _DRIVE_RE.match(normalised):
            raise PathRejected("absolute paths are not allowed")
        parts = [p for p in normalised.split("/") if p not in ("", ".")]
        if ".." in parts:
            raise PathRejected("path traversal ('..') is not allowed")
        if any(":" in part for part in parts):
            raise PathRejected("':' is not allowed in path components")
        if any(part != part.rstrip(". ") and part not in (".", "..") for part in parts):
            raise PathRejected("path components must not end with '.' or a space")
        self._check_policy(parts)
        current = self.root
        for part in parts:
            current = current / part
            if _is_link(current):
                raise PathRejected("symbolic links and junctions are not followed")
        try:
            resolved = current.resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise ToolError(f"not found: {'/'.join(parts) or '.'}") from exc
        if not resolved.is_relative_to(self.root):
            raise PathRejected("path resolves outside the repository root")
        # Re-apply the policy to the names the OS actually resolved (case, 8.3 short names).
        self._check_policy(list(resolved.relative_to(self.root).parts))
        return resolved, "/".join(parts) or "."

    @staticmethod
    def _check_policy(parts: list[str]) -> None:
        for part in parts[:-1]:
            if part.lower() in EXCLUDED_DIRS:
                raise PathRejected(f"'{part}' is excluded by policy")
        if parts and (parts[-1].lower() in EXCLUDED_DIRS or is_secret_name(parts[-1])):
            raise PathRejected(f"'{parts[-1]}' is excluded by policy")

    # -- listing -------------------------------------------------------------------

    def iter_files(self, subdir: str = ".") -> tuple[list[tuple[str, int]], bool]:
        """Sorted (relative path, size) pairs under ``subdir`` and a truncated flag."""
        base, base_rel = self.resolve(subdir)
        if not base.is_dir():
            raise ToolError(f"not a directory: {base_rel}")
        found: list[tuple[str, int]] = []
        truncated = False
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            current = Path(dirpath)
            depth = len(current.relative_to(base).parts)
            dirnames[:] = sorted(
                d
                for d in dirnames
                if d.lower() not in EXCLUDED_DIRS
                and not is_secret_name(d)
                and not _is_link(current / d)
                and depth < MAX_WALK_DEPTH
            )
            for name in sorted(filenames):
                path = current / name
                if is_secret_name(name) or _is_link(path):
                    continue
                if len(found) >= self.limits.max_files:
                    return sorted(found), True
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                found.append((path.relative_to(self.root).as_posix(), size))
        return sorted(found), truncated

    # -- reading -------------------------------------------------------------------

    def read_lines(self, rel: object) -> list[str]:
        """Read a text file as lines, enforcing every sandbox rule."""
        path, norm = self.resolve(rel)
        if norm not in self._files_read and len(self._files_read) >= self.limits.max_files:
            raise LimitExceeded(f"file budget exhausted ({self.limits.max_files} files per worker)")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags)
        except OSError as exc:
            raise ToolError(f"cannot open {norm}") from exc
        try:
            # Check the type on the descriptor before wrapping it: on POSIX, opening a directory
            # read-only succeeds and fdopen() would then raise IsADirectoryError.
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise ToolError(f"not a regular file: {norm}")
            if info.st_size > self.limits.max_file_bytes:
                raise LimitExceeded(
                    f"{norm} is {info.st_size} bytes; limit is {self.limits.max_file_bytes}"
                )
        except BaseException:
            os.close(fd)
            raise
        with os.fdopen(fd, "rb") as handle:
            data = handle.read(self.limits.max_file_bytes + 1)
        if len(data) > self.limits.max_file_bytes:
            raise LimitExceeded(f"{norm} exceeds {self.limits.max_file_bytes} bytes")
        if b"\x00" in data[:8192]:
            raise ToolError(f"{norm} looks like a binary file")
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ToolError(f"{norm} is not valid UTF-8 text") from exc
        self._files_read.add(norm)
        return text.splitlines()

    @property
    def files_read(self) -> int:
        return len(self._files_read)


# -- tool implementations (each returns text for the model) ----------------------------


def format_numbered(path: str, lines: list[str], start: int, end: int, max_chars: int) -> str:
    """Render ``lines[start-1:end]`` as ``  12| text`` with a header, bounded by max_chars."""
    out = [f"# {path} (lines {start}-{end} of {len(lines)})"]
    used = len(out[0])
    for number in range(start, end + 1):
        text = lines[number - 1]
        if len(text) > MAX_LINE_CHARS:
            text = text[:MAX_LINE_CHARS] + "..."
        row = f"{number:>5}| {text}"
        if used + len(row) + 1 > max_chars:
            out.append(f"[output truncated; continue with start_line={number}]")
            break
        out.append(row)
        used += len(row) + 1
    return "\n".join(out)


def list_files(sandbox: Sandbox, subdir: str = ".", suffix: str | None = None) -> str:
    files, truncated = sandbox.iter_files(subdir)
    if suffix:
        files = [(p, s) for p, s in files if p.endswith(suffix)]
    body = "\n".join(f"{path} ({size} bytes)" for path, size in files) or "(no files)"
    note = f"\n[listing truncated at {sandbox.limits.max_files} files]" if truncated else ""
    return f"{len(files)} file(s) under {subdir}:\n{body}{note}"


def read_file(sandbox: Sandbox, path: str, start_line: int = 1, end_line: int | None = None) -> str:
    lines = sandbox.read_lines(path)
    if not isinstance(start_line, int) or isinstance(start_line, bool) or start_line < 1:
        raise ToolError("start_line must be an integer >= 1")
    last = len(lines) if end_line is None else end_line
    if not isinstance(last, int) or isinstance(last, bool) or last < start_line:
        raise ToolError("end_line must be an integer >= start_line")
    last = min(last, len(lines))
    if not lines or start_line > len(lines):
        return f"# {path} (empty or start_line beyond end; file has {len(lines)} lines)"
    _, norm = sandbox.resolve(path)
    return format_numbered(norm, lines, start_line, last, sandbox.limits.max_tool_output_chars)


def find_readme(sandbox: Sandbox) -> str | None:
    for name in sorted(os.listdir(sandbox.root)):
        if name.lower() == "readme.md" and not _is_link(sandbox.root / name):
            return name
    return None


def read_readme(sandbox: Sandbox) -> str:
    name = find_readme(sandbox)
    if name is None:
        return "README.md was not found in the repository root."
    return read_file(sandbox, name)


def project_metadata(sandbox: Sandbox) -> dict[str, object]:
    """Structured facts about the project, parsed (never executed) from pyproject.toml."""
    meta: dict[str, object] = {"pyproject": None}
    try:
        lines = sandbox.read_lines("pyproject.toml")
        data = tomllib.loads("\n".join(lines))
    except ToolError:
        data = None
    except tomllib.TOMLDecodeError:
        meta["pyproject"] = {"error": "pyproject.toml is not valid TOML"}
        data = None
    if data is not None:
        project = data.get("project", {})
        meta["pyproject"] = {
            "name": project.get("name"),
            "version": project.get("version"),
            "requires_python": project.get("requires-python"),
            "dependency_count": len(project.get("dependencies", [])),
            "optional_dependency_groups": sorted(project.get("optional-dependencies", {})),
            "console_scripts": sorted(project.get("scripts", {})),
            "build_backend": data.get("build-system", {}).get("build-backend"),
        }
    files, _ = sandbox.iter_files(".")
    names = {PurePosixPath(p).name.lower() for p, _ in files}
    meta["python_file_count"] = sum(1 for p, _ in files if p.endswith(".py"))
    meta["has_tests_dir"] = any(p.startswith(("tests/", "test/")) for p, _ in files)
    meta["has_license_file"] = any(n.startswith("license") for n in names)
    meta["requirements_files"] = sorted(
        p for p, _ in files if re.fullmatch(r"requirements[^/]*\.txt", PurePosixPath(p).name)
    )
    return meta


def inspect_project_metadata(sandbox: Sandbox) -> str:
    return json.dumps(project_metadata(sandbox), indent=2, sort_keys=True)
