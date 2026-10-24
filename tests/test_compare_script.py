"""The router comparison script runs offline with the mock backend and reports sane metrics."""

from __future__ import annotations

import importlib.util
import json
import sys

import pytest
from conftest import ROOT


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location(
        "compare_routers", ROOT / "scripts" / "compare_routers.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["compare_routers"] = module  # dataclasses look the module up by name
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("compare_routers", None)


def test_dataset_is_well_formed(script):
    rows = script.load_dataset(script.DATASET)
    assert len(rows) >= 30 and len({r.id for r in rows}) == len(rows)
    assert all(r.labels for r in rows)
    assert {"single", "multi", "ambiguous"} <= {r.kind for r in rows}


def test_metrics_math(script):
    from repo_reviewer.models import AgentName as A

    pairs = [
        (frozenset({A.CODE}), frozenset({A.CODE})),
        (frozenset({A.CODE, A.DOCUMENTATION}), frozenset({A.DOCUMENTATION})),
    ]
    m = script.prf(pairs)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == 1.0
    assert m["exact_match"] == 0.5 and m["missed_agent_rate"] == 0.0


def test_baseline_has_full_recall_and_main_runs(script, capsys):
    assert script.main.__module__ == "compare_routers"
    sys.argv = ["compare_routers.py", "--json", "--sweep"]
    assert script.main() == 0
    out = json.loads(capsys.readouterr().out)
    by_name = {s["router"]: s for s in out["summaries"]}
    assert by_name["all-agents (baseline)"]["recall"] == 1.0
    assert "MOCK" in out["banner"]
    assert len(out["sweep"]) == 9
