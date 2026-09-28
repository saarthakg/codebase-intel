"""MCP server: codebase-intel's change-impact analysis as tools for Claude Code,
Cursor and other MCP clients.

The point is to give an agent what it can't cheaply work out itself: which
other files a change usually needs, learned from the repo's git history,
plus which tests to run and who calls the functions it touched. Repos are
found from a path (default: the directory the client started the server in)
and indexed automatically, at HEAD, whenever HEAD moves.

Run over stdio (what MCP clients launch):  codebase-intel mcp

Register with Claude Code (from inside the project to analyze):
  claude mcp add codebase-intel -- codebase-intel mcp
"""
import os
import sys
from pathlib import Path
from typing import Any, Optional

if __package__ in (None, ""):
    # Launched as a script by an MCP client from any working directory.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

from codebase_intel.core.check import check_change as run_check
from codebase_intel.core.impact import analyze_impact
from codebase_intel.core.workspace import NotAGitRepo, ensure_index

# Keep tool results small: they land in the calling agent's context window.
MAX_FILES = 12

# Indexing writes to codebase-intel's own data directory, never to the repo.
TOOL = ToolAnnotations(read_only_hint=True, open_world_hint=False)

mcp = MCPServer(
    name="codebase-intel",
    log_level="WARNING",  # logs go to stderr
    instructions=(
        "Change-impact analysis for the git repository you're working in. Before finishing a "
        "change, call check_change: it reports files that historically change together with the "
        "files you edited but aren't in your change yet, callers of the functions you modified, "
        "and which tests to run. Evidence comes from the repo's git history, import graph and "
        "type-resolved call sites, so it catches coupling that searching the code won't show. "
        "Use impact to ask about one file or symbol before editing it."
    ),
)


class ToolInputError(ToolError):
    """A problem the caller can fix. As a ToolError, its message reaches the
    client; other exceptions are reported as crashes with the text withheld."""


def _repo_path(repo_path: Optional[str]) -> str:
    return os.path.expanduser(repo_path or os.environ.get("CODEBASE_INTEL_REPO") or os.getcwd())


@mcp.tool(annotations=TOOL)
def check_change(repo_path: Optional[str] = None, base: Optional[str] = None, staged: bool = False) -> dict[str, Any]:
    """Check the current change in a git repo: what else it probably needs,
    which tests to run, and who calls the code it modified.

    By default the change is every uncommitted edit (staged, unstaged and new
    files) relative to HEAD. Pass base="main" (or any ref) to check a whole
    branch: everything since its merge base with that ref. `repo_path` is
    any path inside the repo (default: the server's working directory).
    """
    try:
        r = run_check(_repo_path(repo_path), base=base, staged=staged, limit=MAX_FILES)
    except NotAGitRepo as e:
        raise ToolInputError(str(e)) from None
    return {
        "compared_to": r.base,
        "changed_files": r.changed_files,
        "likely_missing": [
            {"file": s.file, "confidence": s.confidence, "why": s.reasons} for s in r.likely_missing
        ],
        "callers_of_changed_code": {
            s.symbol: s.callers_outside_change for s in r.changed_symbols if s.callers_outside_change
        },
        "tests_to_run": r.tests_to_run,
        "not_analyzed": r.not_in_index,
    }


@mcp.tool(annotations=TOOL)
def impact(target: str, repo_path: Optional[str] = None) -> dict[str, Any]:
    """Which files a change to `target` is likely to affect, ranked, with the
    evidence for each. `target` is a file path from the repo root
    ("src/pkg/adapters.py") or a symbol ("HTTPAdapter.send")."""
    try:
        _, repo_id, state = ensure_index(_repo_path(repo_path))
    except NotAGitRepo as e:
        raise ToolInputError(str(e)) from None
    if target not in state.graph.G.nodes and not state.metadata_store.find_symbol(repo_id, target):
        raise ToolInputError(
            f"'{target}' isn't a file or symbol in this repo. File paths are relative to the repo root."
        )
    resp = analyze_impact(target, repo_id, state.graph, state.metadata_store, cochange=state.cochange)
    ranked = resp.high_confidence + resp.medium_confidence + resp.related
    return {
        "target": target,
        "impacted": [
            {"file": f.file_path, "confidence": round(f.confidence, 2), "why": f.reason.split("; ")}
            for f in ranked[:MAX_FILES]
        ],
        "more": max(0, len(ranked) - MAX_FILES),
        "tests_to_run": [t.file_path for t in resp.tests][:MAX_FILES],
    }


def main() -> None:
    mcp.run()  # stdio


if __name__ == "__main__":
    main()
