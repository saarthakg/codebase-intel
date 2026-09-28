"""Check a change: what else probably needs to change, which tests to run,
and who calls the functions it touched.

A change is whatever differs from a base: uncommitted edits (the default),
staged edits, or everything since a branch point (`base="origin/main"`, for
reviewing a branch or PR). The index always holds HEAD (see workspace.py), so
each part of the change is matched to symbols on the side that equals HEAD:
the old side of HEAD → working tree, the new side of merge-base → HEAD.
"""
import subprocess
from typing import Callable, Optional

from pydantic import BaseModel

from codebase_intel.core.definitions import is_test_path
from codebase_intel.core.diff_impact import analyze_symbol_changes, parse_unified_diff, symbols_touched
from codebase_intel.core.workspace import NotAGitRepo, ensure_index, git


class ChangedSymbolReport(BaseModel):
    file: str
    symbol: str
    callers_outside_change: list[str]   # files using it that the change doesn't touch


class Suggestion(BaseModel):
    file: str
    confidence: float
    reasons: list[str]                  # strongest evidence first
    because_of: list[str]               # changed files that led here


class CheckResult(BaseModel):
    repo: str
    head: str
    base: str                           # what the change is measured against
    changed_files: list[str]
    changed_symbols: list[ChangedSymbolReport]
    likely_missing: list[Suggestion]    # files not in the change that likely need to be
    tests_to_run: list[str]
    not_in_index: list[str] = []        # new files, or files the index skips


def _diff(root: str, *args: str) -> str:
    return git(root, "diff", "--no-color", "--no-ext-diff", "-M", *args)


def _untracked(root: str) -> list[str]:
    out = git(root, "ls-files", "--others", "--exclude-standard")
    return [line for line in out.splitlines() if line]


def check_change(
    path: str = ".",
    base: Optional[str] = None,
    staged: bool = False,
    depth: int = 3,
    min_confidence: float = 0.4,
    limit: int = 15,
    progress: Optional[Callable[[str], None]] = None,
) -> CheckResult:
    """Check the change in the repo at `path`.

    base=None: uncommitted edits (staged, unstaged and untracked files).
    base="origin/main" (any ref): everything since the merge base with it,
    committed or not. staged=True: only what's staged, as a pre-commit hook sees it.
    """
    root, repo_id, state = ensure_index(path, progress)
    head = git(root, "rev-parse", "HEAD").strip()
    store, graph = state.metadata_store, state.graph

    parts: list[tuple[str, str]] = []  # (diff text, side that matches the index)
    if base:
        try:
            merge_base = git(root, "merge-base", base, "HEAD").strip()
        except subprocess.CalledProcessError:
            raise NotAGitRepo(f"Can't find a common ancestor of {base!r} and HEAD in {root}.") from None
        parts.append((_diff(root, merge_base, "HEAD"), "new"))
    parts.append((_diff(root, "--cached", "HEAD") if staged else _diff(root, "HEAD"), "old"))

    changed: dict[str, set[str]] = {}
    for text, side in parts:
        for fc in parse_unified_diff(text):
            symbols = changed.setdefault(fc.path, set())
            if fc.path in graph.G.nodes:
                ranges = fc.new_ranges if side == "new" else fc.old_ranges
                rows = store.symbols_in_file(repo_id, fc.path)
                symbols.update(s["qualified_name"] for s in symbols_touched(rows, ranges))
    if not staged:
        for path_ in _untracked(root):
            changed.setdefault(path_, set())

    changed_files = sorted(changed)
    indexed = {f: sorted(changed[f]) for f in changed_files if f in graph.G.nodes}
    not_in_index = [f for f in changed_files if f not in graph.G.nodes]
    base_label = f"{base} (merge base)" if base else ("HEAD, staged changes" if staged else "HEAD, uncommitted changes")
    if not indexed:
        return CheckResult(repo=root, head=head, base=base_label, changed_files=changed_files,
                           changed_symbols=[], likely_missing=[],
                           tests_to_run=[f for f in changed_files if is_test_path(f)],
                           not_in_index=not_in_index)

    impact = analyze_symbol_changes(indexed, repo_id, graph, store, depth, state.cochange)
    in_change = set(changed_files)
    ranked = [f for f in impact.high_confidence + impact.medium_confidence + impact.related
              if f.file_path not in in_change]
    missing = [
        Suggestion(file=f.file_path, confidence=round(f.confidence, 2), reasons=f.reason.split("; "),
                   because_of=f.triggered_by)
        for f in ranked if f.confidence >= min_confidence
    ][:limit]
    tests = [f for f in changed_files if is_test_path(f)]
    tests += [f.file_path for f in ranked if is_test_path(f.file_path) and f.confidence >= min_confidence][:limit]
    return CheckResult(
        repo=root,
        head=head,
        base=base_label,
        changed_files=changed_files,
        changed_symbols=[
            ChangedSymbolReport(file=s.file_path, symbol=s.qualified_name,
                                callers_outside_change=[u for u in s.used_in if u not in in_change])
            for s in impact.changed_symbols
        ],
        likely_missing=missing,
        tests_to_run=list(dict.fromkeys(tests)),
        not_in_index=not_in_index,
    )
