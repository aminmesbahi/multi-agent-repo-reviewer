"""Render a ReviewSummary as Markdown. Output is deterministic (no timings, no timestamps)."""

from __future__ import annotations

from repo_reviewer.models import Mode, ReviewSummary

_MODE_NOTE = {
    Mode.MOCK: "MOCK (scripted checks, no language model was used)",
    Mode.LOCAL: "LOCAL (Ollama via LiteLLM Proxy)",
    Mode.DEEPSEEK: "DEEPSEEK (external API via LiteLLM Proxy; source excerpts left this machine)",
}

_VERIFY_NOTE = {
    "quote": "quote verified",
    "line": "line exists",
    "file": "file exists",
}


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(summary: ReviewSummary) -> str:
    out: list[str] = ["# Repository review", ""]
    out += [
        f"- **Repository:** `{summary.repo}`",
        f"- **Mode:** {_MODE_NOTE[summary.mode]}",
        f"- **Agents run:** {', '.join(w.agent.value for w in summary.workers) or 'none'}",
        f"- **Routing:** `{summary.routing.router}` ({summary.routing.source}) - "
        f"{summary.routing.reason}",
    ]
    if summary.routing.fallback_reason:
        out.append(f"- **Routing fallback:** {_cell(summary.routing.fallback_reason)}")
    for agent, why in sorted(summary.skipped_agents.items()):
        out.append(f"- **Skipped `{agent}`:** {why}")
    out.append("")

    out += [
        "## Workers",
        "",
        "| Agent | Status | Model calls | Tool calls | Findings |",
        "|---|---|---|---|---|",
    ]
    for w in summary.workers:
        status = "ok" if w.status == "ok" else f"FAILED: {_cell(w.error or 'unknown')}"
        out.append(
            f"| {w.agent.value} | {status} | {w.model_calls} | {w.tool_calls} | {len(w.findings)} |"
        )
    out.append("")

    counts = {
        s: sum(1 for f in summary.findings if f.severity.value == s)
        for s in ("high", "medium", "low", "info")
    }
    out += [f"## Findings ({len(summary.findings)})", ""]
    if summary.findings:
        out += [" · ".join(f"{n} {s}" for s, n in counts.items() if n), ""]
    else:
        out += ["No verified findings.", ""]
    for i, f in enumerate(summary.findings, 1):
        location = f"{f.file}:{f.line}" if f.line else f.file
        out += [
            f"### {i}. [{f.severity.value.upper()}] {f.title}",
            "",
            f"- **Agent:** {f.agent.value}"
            + (
                f" (also: {', '.join(a.value for a in f.also_reported_by)})"
                if f.also_reported_by
                else ""
            ),
            f"- **File:** `{location}` ({_VERIFY_NOTE[f.verification]})",
            f"- **Description:** {f.description}",
            f"- **Recommendation:** {f.recommendation}",
        ]
        if f.evidence:
            out.append(f"- **Evidence:** `{f.evidence}`")
        out.append("")

    notes: list[str] = []
    if summary.rejected_findings:
        notes.append(
            f"{summary.rejected_findings} finding(s) were discarded (malformed, or the cited "
            "file/line/quote could not be verified)."
        )
    if summary.duplicates_removed:
        notes.append(f"{summary.duplicates_removed} duplicate finding(s) were merged.")
    if summary.failed_workers:
        notes.append(
            "Some agents failed; this report is partial. Findings from the other agents are "
            "included."
        )
    out += ["## Notes", ""]
    out += [f"- {n}" for n in notes]
    out += [
        "- Verification checks that cited files, lines and quotes exist. It does not prove a "
        "finding is correct; review each one.",
        "- No dependency vulnerability scan was performed.",
        "",
    ]
    return "\n".join(out)
