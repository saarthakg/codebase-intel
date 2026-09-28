#!/usr/bin/env python3
"""Replay a repo's real changes and record the evidence for every candidate
file, for scoring experiments (eval/score_replay.py) and the missing-file
eval.

Walks the main line oldest first. Each merged PR (or direct commit) with
2..max-files indexed files is replayed twice:
  - with one file hidden (once per file): label 1 on the hidden file. Can we
    name the file a change left out?
  - complete, nothing hidden: every suggestion is a false alarm (as far as we
    can tell). How noisy are we on a change that needed nothing more?
Changes are mapped to the symbols their diff touched, as `check` does.
History is rolling: only changes before the one replayed have been counted.

  python eval/build_replay.py --git <full clone> --repo-id requests --out replay_requests.jsonl.gz
"""
import argparse
import gzip
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from codebase_intel.core.diffs import parse_unified_diff, symbols_touched
from codebase_intel.core.features import gather
from codebase_intel.core.history import CoChange, read_history
from codebase_intel.core.ingest import detect_language
from codebase_intel.core.symbols import analyze_file
from codebase_intel.state import get_repo_state


def _git(repo: str, *args: str) -> str:
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def touched_symbols(repo: str, sha: str, path: str, path_then: str) -> list[str]:
    """Qualified names of the innermost symbols `sha` touched in one file
    (against its first parent), from that commit's own versions of the file."""
    language = detect_language(path)
    if language not in ("python", "typescript", "javascript"):
        return []
    diff = _git(repo, "diff", "-U0", "-M", f"{sha}^1", sha, "--", path_then)
    changes = [fc for fc in parse_unified_diff(diff) if fc.path == path_then] or parse_unified_diff(diff)[:1]
    if not changes:
        return []
    names: list[str] = []
    for rev, ranges in ((sha, changes[0].new_ranges), (f"{sha}^1", changes[0].old_ranges)):
        source = _git(repo, "show", f"{rev}:{path_then}")
        if not source or not ranges:
            continue
        rows = [{"qualified_name": s.qualified_name, "start_line": s.start_line, "end_line": s.end_line}
                for s in analyze_file(source, path_then, language).symbols]
        names += [s["qualified_name"] for s in symbols_touched(rows, ranges)]
    return list(dict.fromkeys(names))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--git", required=True, help="Full clone of the indexed repo, at the indexed commit")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--mine", choices=["pr", "commit"], default="pr",
                        help="History counts each merged PR as one change, or each of its commits")
    parser.add_argument("--warmup", type=int, default=300, help="Main-line changes used only as history")
    parser.add_argument("--max-files", type=int, default=10)
    parser.add_argument("--recent", type=int, default=200, help="Changes in the recent-history window")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    state = get_repo_state(args.repo_id)
    indexed = set(state.metadata_store.indexed_files(args.repo_id))
    changes = read_history(args.git, first_parent=True)[::-1]
    commits_by_sha = {c.sha: c for c in read_history(args.git)} if args.mine == "commit" else {}

    history, recent, window = CoChange(), CoChange(), []
    t0, n_queries = time.time(), 0
    with gzip.open(args.out, "wt") as out:
        for i, change in enumerate(changes):
            files = [f for f in dict.fromkeys(change.files) if f in indexed]
            if i >= args.warmup and 2 <= len(files) <= args.max_files and len(change.files) <= 15:
                symbols = {f: touched_symbols(args.git, change.sha, f, change.paths_then.get(f, f)) for f in files}
                for hidden in [None, *files]:
                    query = {f: symbols[f] for f in files if f != hidden}
                    cands = gather(query, args.repo_id, state.graph, state.metadata_store, history, recent=recent)
                    row = {
                        "change": i, "sha": change.sha, "date": change.date, "files": files,
                        "hidden": hidden, "hidden_in_candidates": hidden in cands if hidden else None,
                        "commits_seen": history.commits_used,
                        "candidates": [
                            {k: v for k, v in asdict(c).items() if k not in ("because_of", "uses")}
                            for c in cands.values()
                        ],
                    }
                    out.write(json.dumps(row) + "\n")
                    n_queries += 1
            # Grow history with this change, as one PR or as its own commits
            recent.add(change, keep=indexed)
            window.append(change)
            if len(window) > args.recent:
                recent.add(window.pop(0), keep=indexed, sign=-1)
            if args.mine == "pr" or change.sha not in commits_by_sha and not _is_merge(args.git, change.sha):
                history.add(change, keep=indexed)
            elif change.sha in commits_by_sha:
                history.add(commits_by_sha[change.sha], keep=indexed)
            else:
                for sha in _git(args.git, "rev-list", "--no-merges", f"{change.sha}^1..{change.sha}").split():
                    if sha in commits_by_sha:
                        history.add(commits_by_sha[sha], keep=indexed)
    print(f"{args.repo_id}: {n_queries} replays of {len(changes)} main-line changes, "
          f"mined per {args.mine}, {time.time() - t0:.0f}s -> {args.out}")


_merge_cache: dict[str, bool] = {}


def _is_merge(repo: str, sha: str) -> bool:
    if sha not in _merge_cache:
        _merge_cache[sha] = len(_git(repo, "rev-list", "--parents", "-n", "1", sha).split()) > 2
    return _merge_cache[sha]


if __name__ == "__main__":
    main()
