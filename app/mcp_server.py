"""MCP server: codebase-intel's index as tools for Claude Code, Cursor and other MCP clients.

The client's own model does the reasoning, so there is no LLM call (and no
API cost) here: tools return code, definitions, usages and impact rankings.

Run over stdio (what MCP clients launch):
  /path/to/codebase-intel/.venv/bin/python /path/to/codebase-intel/app/mcp_server.py

Register with Claude Code:
  claude mcp add codebase-intel -- /path/to/.venv/bin/python /path/to/codebase-intel/app/mcp_server.py
"""
import os
import sys
from pathlib import Path
from typing import Any, Literal, Optional

if __package__ in (None, ""):
    # Launched as a script by an MCP client from any working directory.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

import app.core.embeddings as embeddings_module
from app.core import paths
from app.core.definitions import lookup_definition
from app.core.diff_impact import analyze_diff
from app.core.impact import analyze_impact
from app.core.pipeline import IngestError, run_ingestion
from app.core.search import search_chunks
from app.main import _loaded_repos, get_repo_state

# Keep tool results small: they land in the calling agent's context window.
MAX_SNIPPET_CHARS = 1500
MAX_IMPACT_FILES = 25

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)

mcp = MCPServer(
    name="codebase-intel",
    # Logs go to stderr; INFO would include every HTTP request made while
    # loading the embedding model.
    log_level="WARNING",
    instructions=(
        "Code intelligence over locally indexed repositories. Use search_code to find where "
        "something is implemented (natural language or identifiers), find_definition to jump "
        "to a symbol and list every file that uses it, impact to see which files and tests a "
        "change to a file or symbol is likely to affect, and impact_of_diff for an actual "
        "`git diff`. Results cite file paths and line ranges; read those files for full "
        "context. If the repo isn't indexed yet, call ingest_repo first."
    ),
)


class ToolInputError(ToolError):
    """A problem the caller can fix (unknown repo, bad argument). As a ToolError,
    its message reaches the client; other exceptions are reported as crashes
    with the text withheld."""


def _resolve_repo(repo_id: Optional[str]) -> str:
    """repo_id, or the only indexed repo, or CODEBASE_INTEL_REPO_ID."""
    if repo_id:
        return repo_id
    default = os.environ.get("CODEBASE_INTEL_REPO_ID", "").strip()
    if default:
        return default
    known = paths.known_repo_ids()
    if len(known) == 1:
        return known[0]
    if not known:
        raise ToolInputError("No repositories are indexed yet. Call ingest_repo first.")
    raise ToolInputError(f"Several repositories are indexed; pass repo_id, one of: {', '.join(known)}.")


def _state(repo_id: Optional[str]):
    rid = _resolve_repo(repo_id)
    try:
        return rid, get_repo_state(rid)
    except FileNotFoundError:
        raise ToolInputError(
            f"Repository '{rid}' isn't indexed. Indexed: {', '.join(paths.known_repo_ids()) or 'none'}."
        ) from None
    except ValueError as e:  # invalid repo_id format
        raise ToolInputError(str(e)) from None


def _impacted(items, limit: int) -> list[dict[str, Any]]:
    return [
        {"file": f.file_path, "confidence": round(f.confidence, 2), "reason": f.reason}
        for f in items[:limit]
    ]


@mcp.tool(annotations=READ_ONLY)
def list_repos() -> dict[str, Any]:
    """List the repositories that have been indexed, with their size and when they were indexed."""
    import json
    repos = []
    for rid in paths.known_repo_ids():
        meta_file = paths.meta_path(rid)
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        repos.append({
            "repo_id": rid,
            "files": meta.get("files_indexed"),
            "chunks": meta.get("chunks_indexed"),
            "indexed_at": meta.get("ingested_at"),
        })
    return {"repos": repos}


@mcp.tool(annotations=READ_ONLY)
def search_code(
    query: str,
    repo_id: Optional[str] = None,
    top_k: int = 6,
    mode: Literal["semantic", "hybrid", "keyword"] = "semantic",
) -> dict[str, Any]:
    """Find the code that answers a question or matches identifiers.

    Works for natural language ("where are redirects followed?") and for
    identifiers ("get_netrc_auth", "HTTPAdapter.send"). Returns the best
    matching code chunks with file paths and line ranges, best first.
    Use mode="keyword" to match an exact string such as an error message.
    """
    rid, state = _state(repo_id)
    top_k = max(1, min(top_k, 20))
    results = search_chunks(query, rid, top_k, state.faiss_store, state.metadata_store, mode=mode)
    out = []
    for r in results:
        chunk = state.metadata_store.get_chunk(r.chunk_id)
        code = chunk.content if chunk else r.snippet
        out.append({
            "file": r.file_path,
            "lines": f"{r.start_line}-{r.end_line}",
            "code": code if len(code) <= MAX_SNIPPET_CHARS else code[:MAX_SNIPPET_CHARS] + "\n… (truncated)",
        })
    return {"repo_id": rid, "results": out}


@mcp.tool(annotations=READ_ONLY)
def find_definition(symbol: str, repo_id: Optional[str] = None) -> dict[str, Any]:
    """Where a function/class/method is defined, and every file that uses it.

    `symbol` may be bare ("send") or qualified ("HTTPAdapter.send"). For a
    bare name matching several definitions, the best match is returned and
    the rest are listed in other_definitions. Usages are matched by name.
    """
    rid, state = _state(repo_id)
    found = lookup_definition(symbol, state.metadata_store, rid)
    if found is None:
        raise ToolInputError(f"No definition of '{symbol}' in '{rid}'. Try search_code instead.")
    lines_by_file: dict[str, list[int]] = {}
    for loc in found.reference_locations:
        lines_by_file.setdefault(loc.file_path, []).append(loc.line)
    return {
        "symbol": found.qualified_name,
        "kind": found.kind,
        "defined_in": found.defining_file,
        "lines": f"{found.start_line}-{found.end_line}",
        "used_in": [{"file": f, "lines": ls[:20]} for f, ls in sorted(lines_by_file.items())],
        "other_definitions": [
            {"symbol": d.qualified_name, "kind": d.kind, "file": d.file_path, "line": d.start_line}
            for d in found.other_definitions[:10]
        ],
    }


@mcp.tool(annotations=READ_ONLY)
def impact(target: str, repo_id: Optional[str] = None, depth: int = 3) -> dict[str, Any]:
    """Which files (and tests) a change to `target` is likely to affect, ranked.

    `target` is a file path ("src/pkg/adapters.py") or a symbol name. Combines
    the import graph, symbol usages, git co-change history, tests named after
    the file and code similarity; each result says why it's included.
    """
    rid, state = _state(repo_id)
    # A typo'd path would otherwise fall through to "semantically related"
    # guesses for the text itself, which look plausible and mean nothing.
    if target not in state.graph.G.nodes and not state.metadata_store.find_symbol(rid, target):
        raise ToolInputError(
            f"'{target}' isn't a file or symbol in '{rid}'. File paths are relative to the repo "
            f"root; use search_code to find the right name."
        )
    resp = analyze_impact(
        target, rid, state.graph, state.faiss_store, state.metadata_store, embeddings_module,
        depth=depth, cochange=state.cochange,
    )
    ranked = resp.high_confidence + resp.medium_confidence + resp.related
    return {
        "target": target,
        "impacted": _impacted(ranked, MAX_IMPACT_FILES),
        "more": max(0, len(ranked) - MAX_IMPACT_FILES),
        "tests_to_run": [t.file_path for t in resp.tests],
    }


@mcp.tool(annotations=READ_ONLY)
def impact_of_diff(diff: str, repo_id: Optional[str] = None, depth: int = 3) -> dict[str, Any]:
    """Impact of an actual change: pass unified diff text (e.g. `git diff HEAD`).

    Maps changed lines to the functions/classes they touch, lists the files
    using each changed symbol, and ranks everything else likely affected. The
    diff's new side should match the indexed code (re-run ingest_repo first
    if the working tree changed since indexing).
    """
    if not diff.strip():
        raise ToolInputError("diff is empty.")
    rid, state = _state(repo_id)
    resp = analyze_diff(
        diff, rid, state.graph, state.faiss_store, state.metadata_store, embeddings_module,
        depth=depth, cochange=state.cochange,
    )
    ranked = resp.high_confidence + resp.medium_confidence + resp.related
    return {
        "changed_files": resp.targets,
        "changed_symbols": [
            {"file": s.file_path, "symbol": s.qualified_name, "used_in": s.used_in}
            for s in resp.changed_symbols
        ],
        "impacted": _impacted(ranked, MAX_IMPACT_FILES),
        "more": max(0, len(ranked) - MAX_IMPACT_FILES),
        "tests_to_run": [t.file_path for t in resp.tests],
        "not_indexed": resp.unindexed_files,
    }


@mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True,
                                      open_world_hint=False))
def ingest_repo(repo_path: str, repo_id: str) -> dict[str, Any]:
    """Index (or re-index) a local repository so the other tools can query it.

    Safe to repeat: unchanged code isn't re-embedded, so re-indexing after
    edits is fast. `repo_id` is a short name (letters, digits, _ and -).
    """
    try:
        summary = run_ingestion(os.path.expanduser(repo_path), repo_id)
    except (IngestError, ValueError) as e:
        raise ToolInputError(str(e)) from None
    _loaded_repos.pop(repo_id, None)  # drop any stale in-memory state for this repo
    return {k: summary[k] for k in (
        "repo_id", "files_indexed", "chunks_indexed", "symbols_extracted", "edges_in_graph",
        "files_skipped", "chunks_embedded", "chunks_reused",
    )}


def main() -> None:
    mcp.run()  # stdio


if __name__ == "__main__":
    main()
