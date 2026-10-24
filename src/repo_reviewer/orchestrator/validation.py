"""Final-stage checks on findings: verify against the files, deduplicate, sort.

Schema validation (done earlier by Pydantic) only proves a finding is well-formed. The checks
here prove that the cited file exists, the cited line exists and the quoted evidence really
appears next to it. They do NOT prove the finding is *correct*: a model can quote a real line
and still draw a wrong conclusion. The report says so.
"""

from __future__ import annotations

import logging

from repo_reviewer.models import Finding, Limits, WorkerResult
from repo_reviewer.tools.filesystem import Sandbox, ToolError

log = logging.getLogger("repo_reviewer.validation")
EVIDENCE_WINDOW = 2  # lines of slack either side of the cited line


def _squash(text: str) -> str:
    return " ".join(text.split())


def verify_finding(finding: Finding, sandbox: Sandbox) -> Finding | None:
    """Return the finding with its verification level set, or None if it cannot be verified."""
    try:
        lines = sandbox.read_lines(finding.file)
    except ToolError as exc:
        log.info("rejected finding (%s): %s", finding.file, exc)
        return None
    if finding.line is not None and finding.line > len(lines):
        log.info("rejected finding: line %d beyond end of %s", finding.line, finding.file)
        return None
    level = "file"
    if finding.line is not None:
        level = "line"
    if finding.evidence:
        if finding.line is None:
            window = lines
        else:
            low = max(0, finding.line - 1 - EVIDENCE_WINDOW)
            window = lines[low : finding.line + EVIDENCE_WINDOW]
        if _squash(finding.evidence) not in _squash("\n".join(window)):
            log.info("rejected finding: evidence not found near %s:%s", finding.file, finding.line)
            return None
        level = "quote"
    return finding.model_copy(update={"verification": level}, deep=True)


def _dedupe_key(finding: Finding) -> tuple[str, int, str]:
    return (finding.file.lower(), finding.line or 0, _squash(finding.title.lower()))


def deduplicate(findings: list[Finding]) -> tuple[list[Finding], int]:
    """Merge findings with the same file, line and title. Highest severity wins."""
    merged: dict[tuple[str, int, str], Finding] = {}
    removed = 0
    for finding in sorted(findings, key=_sort_key):
        key = _dedupe_key(finding)
        kept = merged.get(key)
        if kept is None:
            merged[key] = finding
            continue
        removed += 1
        if finding.agent != kept.agent and finding.agent not in kept.also_reported_by:
            kept.also_reported_by.append(finding.agent)
    return list(merged.values()), removed


def _sort_key(finding: Finding) -> tuple[int, str, int, str, str]:
    return (
        -finding.severity.rank,
        finding.file.lower(),
        finding.line or 0,
        finding.agent.value,
        finding.title.lower(),
    )


def finalize_findings(
    results: list[WorkerResult], repo_root: str, limits: Limits
) -> tuple[list[Finding], int, int]:
    """(verified, deduplicated, sorted findings; rejected count; duplicates removed).

    Only ``ok`` workers contribute. Failed workers are reported separately and never
    discard the findings of the others.
    """
    sandbox = Sandbox(repo_root, limits.model_copy(update={"max_files": 1_000}))
    rejected = sum(r.rejected_findings for r in results)
    verified: list[Finding] = []
    for result in results:
        if result.status != "ok":
            continue
        for finding in result.findings:
            checked = verify_finding(finding, sandbox)
            if checked is None:
                rejected += 1
            else:
                verified.append(checked)
    unique, removed = deduplicate(verified)
    return sorted(unique, key=_sort_key), rejected, removed
