#!/usr/bin/env python3
"""Score impact analysis against real commits: "if I change file A, which other
files will this change also need to touch?"

For every commit after --cutoff that changed 2–15 Python files that still exist,
each changed *source* file is used as the query, and the commit's other changed
files (source and tests) are the answer. Impact analysis ranks candidate files;
we measure how many of the real co-changed files it ranks near the top.

No leakage from the future into the co-change signal: it is built only from
commits up to --cutoff. (The import graph and index are today's; the code's
structure changes slowly, but this is a mild optimistic bias on graph signals.)

Needs a full (non-shallow) clone checked out at the indexed commit:
  git clone https://github.com/psf/requests requests-full
  git -C requests-full checkout <repo_commit from eval/requests_bench.yaml>
  python eval/run_history_eval.py --git requests-full --repo-id requests

Dev/held-out split: tune on --dev-until (default commits 2019-01-01..2022-12-31),
report the held-out years (2023+) separately and never tune on them.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.definitions import is_test_path
import subprocess

from app.core.diff_impact import analyze_symbol_changes, parse_unified_diff, symbols_touched
from app.core.history import CoChange, read_history
from app.core.impact import analyze_impact
from app.core.symbols import analyze_file
from app.state import get_repo_state

KS = (5, 10)


def ranked_files(response) -> list[str]:
    items = response.high_confidence + response.medium_confidence + response.related
    return [i.file_path for i in items]


def _git(repo: str, *args: str) -> str:
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def changed_symbols(repo: str, sha: str, path_then: str) -> list[str]:
    """Qualified names of the innermost symbols a commit touched in one file,
    using that commit's own versions of the file (line numbers match the diff)."""
    diff = _git(repo, "show", "-U0", "--format=", "-M", sha, "--", path_then)
    changes = [fc for fc in parse_unified_diff(diff) if fc.path == path_then] or parse_unified_diff(diff)[:1]
    if not changes:
        return []
    fc = changes[0]
    names: list[str] = []
    for rev, ranges in ((sha, fc.new_ranges), (f"{sha}^", fc.old_ranges)):
        source = _git(repo, "show", f"{rev}:{path_then}")
        if not source or not ranges:
            continue
        rows = [
            {"qualified_name": s.qualified_name, "start_line": s.start_line, "end_line": s.end_line}
            for s in analyze_file(source, path_then, "python").symbols
        ]
        names += [s["qualified_name"] for s in symbols_touched(rows, ranges)]
    return list(dict.fromkeys(names))


def score(cases: list[tuple[str, set[str], list[str]]]) -> dict:
    """cases: (query, true co-changed files, ranking)."""
    out: dict[str, float] = {}
    for k in KS:
        out[f"recall@{k}"] = sum(len(set(r[:k]) & t) / len(t) for _, t, r in cases) / len(cases)
        tests = [(q, {f for f in t if is_test_path(f)}, r) for q, t, r in cases]
        tests = [(q, t, r) for q, t, r in tests if t]
        out[f"test_recall@{k}"] = (
            sum(len(set(r[:k]) & t) / len(t) for _, t, r in tests) / len(tests) if tests else 0.0
        )
    rr = []
    for _, t, r in cases:
        rank = next((i for i, f in enumerate(r, 1) if f in t), None)
        rr.append(1.0 / rank if rank else 0.0)
    out["mrr"] = sum(rr) / len(rr)
    out["n_queries"] = len(cases)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--git", required=True, help="Full clone of the indexed repo, at the indexed commit")
    parser.add_argument("--repo-id", default="requests")
    parser.add_argument("--cutoff", default="2018-12-31", help="Co-change uses commits up to this date")
    parser.add_argument("--dev-until", default="2022-12-31", help="Test commits up to here are the dev set")
    parser.add_argument("--no-cochange", action="store_true", help="Score without the co-change signal")
    parser.add_argument("--diff-level", action="store_true",
                        help="Use each commit's diff (changed symbols) instead of just the file")
    parser.add_argument("--out", help="Write results JSON here")
    args = parser.parse_args()

    state = get_repo_state(args.repo_id)
    indexed = set(state.graph.G.nodes)
    history = read_history(args.git)
    train = [c for c in history if c.date <= args.cutoff]
    cochange = None if args.no_cochange else CoChange.from_commits(train, keep=indexed)

    splits: dict[str, list] = {"dev": [], "heldout": []}
    for commit in history:
        if commit.date <= args.cutoff:
            continue
        changed = [f for f in commit.files if f.endswith(".py") and f in indexed]
        if not 2 <= len(changed) <= 15 or len(commit.files) > 30:
            continue
        split = "dev" if commit.date <= args.dev_until else "heldout"
        for query in changed:
            if is_test_path(query):
                continue
            truth = set(changed) - {query}
            if args.diff_level:
                symbols = changed_symbols(args.git, commit.sha, commit.paths_then.get(query, query))
                response = analyze_symbol_changes(
                    {query: symbols}, args.repo_id, state.graph,
                    state.metadata_store, depth=3, cochange=cochange,
                )
            else:
                response = analyze_impact(
                    query, args.repo_id, state.graph, state.metadata_store,
                    depth=3, cochange=cochange,
                )
            splits[split].append((query, truth, ranked_files(response)))

    results = {name: score(cases) for name, cases in splits.items() if cases}
    label = ("without co-change" if args.no_cochange else "with co-change") + (", diff-level" if args.diff_level else "")
    print(f"\nHistory eval ({label}); co-change from commits ≤ {args.cutoff}, "
          f"{cochange.commits_used if cochange else 0} commits used\n")
    for name, metrics in results.items():
        cells = "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
        print(f"  {name:<8} {cells}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
