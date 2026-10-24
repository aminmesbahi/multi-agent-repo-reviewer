# Contributing

Thanks for helping. The project is intentionally small and educational: please keep changes
easy to read.

## Setup

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Everything above must pass without Docker, a model download or any API key. Tests that would
need those do not belong in the default suite.

## Ground rules

- **Read-only.** No tool may write, execute or install anything from the reviewed repository.
  All file access goes through `tools/filesystem.py::Sandbox`.
- **No hidden cloud use.** Every real model call goes through the LiteLLM Proxy; cloud models must
  be an explicit `--mode`. Do not add provider SDKs or silent fallbacks.
- **No fabricated results.** Do not paste invented output into docs, and say plainly what you did
  not verify.
- **Tests exercise the real graph** (`tests/test_graph.py`, `tests/test_e2e_mock.py`). Mock the
  model, not the orchestration.
- Keep the architecture small. A new abstraction needs a concrete second user.
- No secrets, employer-related content, internal URLs or proprietary code in samples or tests.
- Match the surrounding style; Ruff enforces formatting and the lint set in `pyproject.toml`.

## Pull requests

1. Describe the behaviour change and how you tested it.
2. Update `README.md` / `docs/` when a CLI option, env var or behaviour changes; every command in
   the docs must work as written.
3. Add or update tests, including a failing-case test for security or limit changes.

## Reporting security issues

See [SECURITY.md](SECURITY.md); please do not open a public issue for vulnerabilities.
