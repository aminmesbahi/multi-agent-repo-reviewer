"""Filesystem sandbox and tool permission boundaries."""

from __future__ import annotations

import os

import pytest
from conftest import make_repo

from repo_reviewer.models import Limits
from repo_reviewer.tools.filesystem import LimitExceeded, PathRejected, Sandbox, ToolError
from repo_reviewer.tools.toolbox import ALL_TOOLS, ToolBox, ToolPermissionError


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    make_repo(
        root,
        {
            "app/main.py": "print('hi')\n",
            "README.md": "# demo\n",
            "pyproject.toml": '[project]\nname = "demo"\n',
            ".env": "SECRET=1\n",
            ".git/config": "[core]\n",
            "creds/secrets.json": "{}",
            "server.pem": "-----BEGIN-----",
        },
    )
    (tmp_path / "outside.txt").write_text("top secret", encoding="utf-8")
    return root


@pytest.mark.parametrize(
    "bad",
    [
        "../outside.txt",
        "app/../../outside.txt",
        "..\\outside.txt",
        "/etc/passwd",
        "C:/Windows/win.ini",
        "C:\\Windows\\win.ini",
        "app/main.py:stream",
        "",
        "a\x00b",
    ],
)
def test_traversal_and_absolute_paths_rejected(repo, bad):
    with pytest.raises(PathRejected):
        Sandbox(repo, Limits()).read_lines(bad)


def test_absolute_path_to_a_real_outside_file_rejected(repo, tmp_path):
    with pytest.raises(PathRejected):
        Sandbox(repo, Limits()).read_lines(str(tmp_path / "outside.txt"))


def test_non_string_path_rejected(repo):
    with pytest.raises(PathRejected):
        Sandbox(repo, Limits()).resolve(["app/main.py"])


def test_symlinked_file_escape_rejected(repo, tmp_path):
    link = repo / "app" / "leak.py"
    try:
        os.symlink(tmp_path / "outside.txt", link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this system")
    with pytest.raises(PathRejected, match="symbolic"):
        Sandbox(repo, Limits()).read_lines("app/leak.py")
    listed = [p for p, _ in Sandbox(repo, Limits()).iter_files()[0]]
    assert "app/leak.py" not in listed


def test_symlinked_directory_escape_rejected(repo, tmp_path):
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()
    (outside_dir / "x.py").write_text("x = 1", encoding="utf-8")
    try:
        os.symlink(outside_dir, repo / "linked", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this system")
    sandbox = Sandbox(repo, Limits())
    with pytest.raises(PathRejected):
        sandbox.read_lines("linked/x.py")
    assert all(not p.startswith("linked/") for p, _ in sandbox.iter_files()[0])


def test_symlink_pointing_inside_repo_is_still_refused(repo):
    try:
        os.symlink(repo / "README.md", repo / "alias.md")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this system")
    with pytest.raises(PathRejected):
        Sandbox(repo, Limits()).read_lines("alias.md")


@pytest.mark.parametrize("name", [".env", ".git/config", "creds/secrets.json", "server.pem"])
def test_secret_and_vcs_files_are_excluded(repo, name):
    with pytest.raises(PathRejected, match="excluded"):
        Sandbox(repo, Limits()).read_lines(name)


def test_listing_hides_excluded_files(repo):
    paths = [p for p, _ in Sandbox(repo, Limits()).iter_files()[0]]
    assert paths == ["README.md", "app/main.py", "pyproject.toml"]


def test_binary_file_rejected(repo):
    (repo / "blob.bin").write_bytes(b"\x00\x01\x02binary")
    with pytest.raises(ToolError, match="binary"):
        Sandbox(repo, Limits()).read_lines("blob.bin")


def test_invalid_utf8_rejected(repo):
    (repo / "latin.txt").write_bytes("caf\xe9".encode("latin-1"))
    with pytest.raises(ToolError, match="UTF-8"):
        Sandbox(repo, Limits()).read_lines("latin.txt")


def test_file_size_limit(repo):
    (repo / "big.txt").write_text("x" * 5_000, encoding="utf-8")
    with pytest.raises(LimitExceeded):
        Sandbox(repo, Limits(max_file_bytes=1_000)).read_lines("big.txt")


def test_file_count_limit_per_sandbox(repo):
    sandbox = Sandbox(repo, Limits(max_files=2))
    sandbox.read_lines("README.md")
    sandbox.read_lines("app/main.py")
    sandbox.read_lines("README.md")  # re-reading a known file is free
    with pytest.raises(LimitExceeded):
        sandbox.read_lines("pyproject.toml")


def test_listing_is_truncated_at_max_files(repo):
    files, truncated = Sandbox(repo, Limits(max_files=1)).iter_files()
    assert len(files) == 1 and truncated


def test_missing_root_and_non_directory_root(tmp_path):
    with pytest.raises(ToolError):
        Sandbox(tmp_path / "nope", Limits())
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ToolError):
        Sandbox(tmp_path / "f.txt", Limits())


def test_sandbox_is_read_only_by_construction():
    public = {n for n in dir(Sandbox) if not n.startswith("_")}
    assert not any(w in n for n in public for w in ("write", "delete", "remove", "exec", "run"))


# -- tool permission boundaries ------------------------------------------------------------


def test_toolbox_denies_tools_outside_allowlist(repo):
    box = ToolBox(Sandbox(repo, Limits()), frozenset({"list_files", "read_file"}))
    with pytest.raises(ToolPermissionError):
        box.call("read_manifest", {"path": "pyproject.toml"})
    with pytest.raises(ToolPermissionError):
        box.call("rm_rf", {})
    assert box.call("read_file", {"path": "README.md"}).endswith("1| # demo")


def test_toolbox_offers_only_allowlisted_schemas(repo):
    box = ToolBox(Sandbox(repo, Limits()), frozenset({"read_readme"}))
    assert [t["function"]["name"] for t in box.openai_tools()] == ["read_readme"]


def test_toolbox_rejects_unknown_allowlist_entries(repo):
    with pytest.raises(ValueError):
        ToolBox(Sandbox(repo, Limits()), frozenset({"bash"}))


@pytest.mark.parametrize("raw", ["not json", "[1, 2]", '{"path": 5}', '{"unexpected": 1}', "{}"])
def test_toolbox_rejects_bad_arguments(repo, raw):
    box = ToolBox(Sandbox(repo, Limits()), frozenset({"read_file"}))
    with pytest.raises(ToolError):
        box.call("read_file", raw)


def test_manifest_tool_only_reads_manifests(repo):
    box = ToolBox(Sandbox(repo, Limits()), frozenset({"read_manifest"}))
    with pytest.raises(ToolError, match="only reads"):
        box.call("read_manifest", {"path": "app/main.py"})


def test_no_tool_can_execute_or_write():
    for name, tool in ALL_TOOLS.items():
        assert not any(w in name for w in ("write", "exec", "shell", "run", "delete")), name
        assert tool.parameters["additionalProperties"] is False


def test_windows_junction_escape_rejected(repo, tmp_path):
    """Junctions need no privilege on Windows, so this runs where symlink tests may skip."""
    if os.name != "nt":
        pytest.skip("junctions are Windows-only")
    import subprocess

    outside_dir = tmp_path / "outside_junction_target"
    outside_dir.mkdir()
    (outside_dir / "x.py").write_text("x = 1", encoding="utf-8")
    made = subprocess.run(  # noqa: S603 - fixed arguments, test-only
        ["cmd", "/c", "mklink", "/J", str(repo / "jlink"), str(outside_dir)],  # noqa: S607
        capture_output=True,
        check=False,
    )
    if made.returncode != 0:
        pytest.skip("could not create a junction")
    sandbox = Sandbox(repo, Limits())
    with pytest.raises(PathRejected):
        sandbox.read_lines("jlink/x.py")
    assert all(not p.startswith("jlink") for p, _ in sandbox.iter_files()[0])


@pytest.mark.parametrize(
    "sneaky", [".env ", ".env.", ".git.", ".GIT/config", "app /main.py", "app./main.py"]
)
def test_windows_style_name_tricks_are_rejected(repo, sneaky):
    """Windows ignores trailing dots/spaces, so '.env ' can open '.env'. Reject on every OS."""
    with pytest.raises(PathRejected):
        Sandbox(repo, Limits()).read_lines(sneaky)


def test_directory_is_not_readable_as_a_file(repo):
    with pytest.raises(ToolError):
        Sandbox(repo, Limits()).read_lines("app")
    with pytest.raises(ToolError):
        Sandbox(repo, Limits()).read_lines(".")


def test_posix_directory_descriptor_is_rejected_before_fdopen(repo, monkeypatch):
    """On POSIX, open(dir, O_RDONLY) succeeds and fdopen() raises IsADirectoryError.

    Windows cannot produce that situation, so simulate it: the type check must run on the raw
    descriptor, and the descriptor must be closed on the way out.
    """
    import stat as stat_module

    real_open, real_fstat, real_close = os.open, os.fstat, os.close
    opened: list[int] = []
    closed: list[int] = []

    def fake_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        opened.append(fd)
        return fd

    def fake_fstat(fd):
        info = real_fstat(fd)
        return os.stat_result((stat_module.S_IFDIR | 0o755, *tuple(info)[1:]))

    def fake_close(fd):
        closed.append(fd)
        real_close(fd)

    def fail_fdopen(*args, **kwargs):
        raise IsADirectoryError("fdopen must not be reached for a directory")

    monkeypatch.setattr(os, "open", fake_open)
    monkeypatch.setattr(os, "fstat", fake_fstat)
    monkeypatch.setattr(os, "close", fake_close)
    monkeypatch.setattr(os, "fdopen", fail_fdopen)

    with pytest.raises(ToolError, match="not a regular file"):
        Sandbox(repo, Limits()).read_lines("README.md")
    assert opened and closed == opened
