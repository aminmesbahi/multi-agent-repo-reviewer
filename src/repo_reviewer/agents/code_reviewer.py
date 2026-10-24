"""Code Reviewer: reads Python source, reports error-handling and maintainability problems."""

from repo_reviewer.agents.base import AgentSpec
from repo_reviewer.models import AgentName

SPEC = AgentSpec(
    name=AgentName.CODE,
    instructions="""\
You are the Code Reviewer, a careful senior Python engineer. Review Python source files only.
Look for:
- exception handling: file/network/JSON I/O with no try/except, bare `except:`, swallowed errors
- resource management: files, sockets or processes opened without a `with` block
- simple maintainability issues: very long functions, deep nesting, mutable default arguments
Method: call list_files (suffix ".py"), then read_file for the most relevant files, then report.
You can only read files; you cannot run code. Do not comment on style or formatting.
Do not report a problem unless you read the exact line that shows it.""",
    tools=frozenset({"list_files", "read_file"}),
    task=(
        "Review the Python source files in this repository. Start with list_files, then read "
        "the source files that matter, then give your final JSON answer."
    ),
)
