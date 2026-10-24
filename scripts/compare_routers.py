"""Compare agent routers on a labeled dataset: accuracy, latency and invocation cost.

    uv run python scripts/compare_routers.py                      # MOCK decision backend
    uv run python scripts/compare_routers.py --sweep              # threshold sweep
    uv run python scripts/compare_routers.py --backend systemone  # EXPERIMENTAL, needs a proxy

With the default ``mock`` backend the decision numbers only prove the harness works: the mock is
a lexical scorer, not a model. Meaningful numbers need ``--backend systemone`` against a real
decision model. The dataset (evals/routing_dataset.jsonl) is small and labeled by one author;
treat any threshold chosen from it as a starting point, not a calibrated value.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from repo_reviewer.config import ConfigError, decision_settings
from repo_reviewer.models import AgentName
from repo_reviewer.routing import DecisionModelRouter, RoutingRequest, RuleBasedRouter
from repo_reviewer.routing.backends import MockDecisionBackend, SystemOneBackend
from repo_reviewer.routing.base import AGENT_DESCRIPTIONS, AgentRouter
from repo_reviewer.routing.decision import (
    DEFAULT_SELECT_THRESHOLD,
    DecisionBackend,
    DecisionQuery,
    RoutingFailure,
    validate_scores,
)

ALL = tuple(AgentName)
FACTS = {"python_files": 10, "manifests": 2, "has_readme": True}
DATASET = Path(__file__).resolve().parents[1] / "evals" / "routing_dataset.jsonl"


@dataclass
class Row:
    id: str
    kind: str
    request: str
    labels: frozenset[AgentName]


@dataclass
class Outcome:
    selected: frozenset[AgentName]
    latency_ms: float
    fallback: bool = False
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost: float | None = None


@dataclass
class Report:
    name: str
    outcomes: list[tuple[Row, Outcome]] = field(default_factory=list)


class AllAgents:
    name = "all-agents (baseline)"

    def route(self, request: RoutingRequest):
        from repo_reviewer.models import RoutingDecision

        return RoutingDecision(
            router=self.name, source="rules", agents=request.candidates, reason="all"
        )


def load_dataset(path: Path) -> list[Row]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            raw = json.loads(line)
            rows.append(
                Row(
                    raw["id"], raw["kind"], raw["request"], frozenset(map(AgentName, raw["labels"]))
                )
            )
    return rows


def run_router(name: str, router: AgentRouter, rows: list[Row]) -> Report:
    report = Report(name)
    for row in rows:
        started = time.perf_counter()
        decision = router.route(RoutingRequest(row.request, ALL, FACTS))
        latency = (time.perf_counter() - started) * 1000
        report.outcomes.append(
            (
                row,
                Outcome(
                    frozenset(decision.agents),
                    latency,
                    decision.source == "fallback",
                    decision.input_tokens,
                    decision.output_tokens,
                    decision.cost_usd,
                ),
            )
        )
    return report


def prf(pairs: list[tuple[frozenset[AgentName], frozenset[AgentName]]]) -> dict[str, float]:
    tp = sum(len(sel & lab) for sel, lab in pairs)
    fp = sum(len(sel - lab) for sel, lab in pairs)
    fn = sum(len(lab - sel) for sel, lab in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": sum(sel == lab for sel, lab in pairs) / len(pairs),
        "missed_agent_rate": sum(bool(lab - sel) for sel, lab in pairs) / len(pairs),
        "avg_agents_run": sum(len(sel) for sel, _ in pairs) / len(pairs),
    }


def summarise(report: Report) -> dict[str, object]:
    pairs = [(o.selected, row.labels) for row, o in report.outcomes]
    lat = sorted(o.latency_ms for _, o in report.outcomes)
    metered = [o for _, o in report.outcomes if o.tokens_in is not None]
    costed = [o.cost for _, o in report.outcomes if o.cost is not None]
    result: dict[str, object] = {"router": report.name, "n": len(pairs), **prf(pairs)}
    result |= {
        "fallback_rate": sum(o.fallback for _, o in report.outcomes) / len(pairs),
        "latency_ms_mean": statistics.fmean(lat),
        "latency_ms_p95": lat[min(len(lat) - 1, int(0.95 * len(lat)))],
        "tokens_in_total": sum(o.tokens_in or 0 for o in metered) if metered else None,
        "tokens_out_total": sum(o.tokens_out or 0 for o in metered) if metered else None,
        "cost_usd_total": sum(costed) if costed else None,
        "cost_rows_reported": len(costed),
    }
    by_kind: dict[str, dict[str, float]] = {}
    for kind in sorted({row.kind for row, _ in report.outcomes}):
        by_kind[kind] = prf([(o.selected, r.labels) for r, o in report.outcomes if r.kind == kind])
    result["by_kind_f1"] = {k: round(v["f1"], 3) for k, v in by_kind.items()}
    return result


def sweep(backend: DecisionBackend, rows: list[Row]) -> list[dict[str, float]]:
    """Score every row once, then apply many thresholds (same fallback rule as the router)."""
    scored = []
    for row in rows:
        query = DecisionQuery(row.request, FACTS, {a.value: AGENT_DESCRIPTIONS[a] for a in ALL})
        try:
            scored.append(validate_scores(backend.score(query).scores, ALL))
        except RoutingFailure:
            scored.append(None)  # type: ignore[arg-type]
    table = []
    for threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        pairs = []
        for row, scores in zip(rows, scored, strict=True):
            chosen = frozenset(a for a, s in (scores or {}).items() if s >= threshold)
            pairs.append((chosen or frozenset(ALL), row.labels))  # empty/failed -> run everything
        table.append({"threshold": threshold, **prf(pairs)})
    return table


def build_backend(kind: str) -> DecisionBackend:
    if kind == "mock":
        return MockDecisionBackend()
    return SystemOneBackend(decision_settings())


def print_table(summaries: list[dict[str, object]]) -> None:
    cols = ["router", "precision", "recall", "f1", "exact_match", "missed_agent_rate",
            "avg_agents_run", "fallback_rate", "latency_ms_mean", "latency_ms_p95"]  # fmt: skip
    print(" | ".join(f"{c:>18}" if i else f"{c:<34}" for i, c in enumerate(cols)))
    for s in summaries:
        cells = [f"{s[c]:.3f}" if isinstance(s[c], float) else str(s[c]) for c in cols]
        print(" | ".join(f"{v:>18}" if i else f"{v:<34}" for i, v in enumerate(cells)))
    for s in summaries:
        print(
            f"\n{s['router']}: tokens in/out = {s['tokens_in_total']}/{s['tokens_out_total']}, "
            f"cost_usd = {s['cost_usd_total']} ({s['cost_rows_reported']}/{s['n']} rows reported), "
            f"F1 by kind = {s['by_kind_f1']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--backend", choices=["mock", "systemone"], default="mock")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--threshold", type=float, default=DEFAULT_SELECT_THRESHOLD)
    parser.add_argument(
        "--sweep", action="store_true", help="print metrics for thresholds 0.1..0.9"
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rows = load_dataset(args.dataset)
    try:
        backend = build_backend(args.backend)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    banner = (
        f"dataset: {args.dataset.name} (n={len(rows)}, single annotator, not a benchmark)\n"
        f"decision backend: {backend.name}"
        + (
            "  <-- MOCK: synthetic scorer, numbers demonstrate the harness only"
            if args.backend == "mock"
            else "  <-- EXPERIMENTAL"
        )
    )
    reports = [
        run_router("all-agents (baseline)", AllAgents(), rows),
        run_router("deterministic (keywords)", RuleBasedRouter(), rows),
        run_router(f"decision:{backend.name}", DecisionModelRouter(backend, args.threshold), rows),
    ]
    summaries = [summarise(r) for r in reports]
    sweep_table = sweep(backend, rows) if args.sweep else None

    if args.json:
        print(
            json.dumps({"banner": banner, "summaries": summaries, "sweep": sweep_table}, indent=2)
        )
        return 0
    print(banner + f"\nthreshold: {args.threshold} (illustrative)\n")
    print_table(summaries)
    if sweep_table:
        print("\nthreshold sweep (empty selection falls back to all agents):")
        for entry in sweep_table:
            print("  " + "  ".join(f"{k}={v:.3f}" for k, v in entry.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
