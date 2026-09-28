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
from codebase_intel.core.diffs import parse_unified_diff, symbols_touched
from codebase_intel.core.predict import predict
from codebase_intel.core.workspace import NotAGitRepo, ensure_index, git

# Show a file as likely missing at this probability. The model is calibrated:
# on repos it wasn't trained on, 31% of files shown at >= 0.2 were really
# missing, with ~0.6 false warnings per change that needed nothing more
# (eval/results/model_loro.json).
DEFAULT_MIN_CONFIDENCE = 0.2
ALSO_CONSIDER = 0.1


class ChangedSymbolReport(BaseModel):
    file: str
    symbol: str
    callers_outside_change: list[str]   # files using it that the change doesn't touch


class Suggestion(BaseModel):
    file: str
    confidence: float                   # calibrated probability that the change needs this file
    reasons: list[str]                  # strongest evidence first
    because_of: list[str]               # changed files that led here


class CheckResult(BaseModel):
    repo: str
    head: str
    base: str                           # what the change is measured against
    changed_files: list[str]
    changed_symbols: list[ChangedSymbolReport]
    likely_missing: list[Suggestion]    # files not in the change that likely need to be
    also_consider: list[Suggestion] = []  # less likely, still worth a look
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
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
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

    in_change = set(changed_files)
    predictions = [p for p in predict(indexed, repo_id, state, depth) if p.file not in in_change]

    def suggestion(p) -> Suggestion:
        return Suggestion(file=p.file, confidence=round(p.probability, 2), reasons=p.reasons,
                          because_of=p.because_of)

    missing = [suggestion(p) for p in predictions if p.probability >= min_confidence][:limit]
    also = [suggestion(p) for p in predictions if min(min_confidence, ALSO_CONSIDER) <= p.probability < min_confidence][:5]

    callers: dict[tuple[str, str], list[str]] = {(f, sym): [] for f, syms in indexed.items() for sym in syms}
    for p in predictions:
        for key in p.uses:
            callers.setdefault(tuple(key), []).append(p.file)

    # Tests: the ones changed, then every test with a link to the change
    # (named after it, calling it, importing it, changing with it), most likely first.
    tests = [f for f in changed_files if is_test_path(f)]
    tests += [p.file for p in predictions if p.kind == "test" and (p.probability >= 0.05 or p.uses or p.reasons)]
    return CheckResult(
        repo=root,
        head=head,
        base=base_label,
        changed_files=changed_files,
        changed_symbols=[
            ChangedSymbolReport(file=f, symbol=sym, callers_outside_change=sorted(users))
            for (f, sym), users in callers.items()
        ],
        likely_missing=missing,
        also_consider=also,
        tests_to_run=list(dict.fromkeys(tests)),
        not_in_index=not_in_index,
    )
