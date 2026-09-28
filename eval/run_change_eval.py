#!/usr/bin/env python3
"""Missing-file eval: given most of a real change, does impact name the rest?

The job this tool exists for is catching the file a change forgot. Replay
each merged change on the main line (a PR's whole diff, or a direct commit),
hide one of its files, and ask impact about the others. A hit is the hidden
file ranking in the top k. This is the "incomplete transaction" evaluation of
change-recommendation research (Zimmermann et al., TSE 2005), with exact
labels on thousands of real changes.

History is rolling, as in use: each change is scored with co-change mined
from the main-line changes before it, never after.

  python eval/run_change_eval.py --git <full clone> --repo-id requests
"""
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.history import CoChange, read_history
from app.core.impact import analyze_impact_batch
from app.state import get_repo_state

KS = (1, 3, 5, 10)


def kind(path: str) -> str:
    if re.search(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*$|_test\.py$|\.test\.[jt]sx?$", path):
        return "test"
    if path.endswith((".py", ".ts", ".tsx", ".js", ".jsx")):
        return "source"
    if re.search(r"\.(rst|md|txt)$", path) or path.startswith("docs/"):
        return "docs"
    return "other"


def ranked(response) -> list[str]:
    return [f.file_path for f in response.high_confidence + response.medium_confidence + response.related]


def summarize(ranks: list) -> dict:
    """ranks: 1-based rank of the hidden file, or None if not listed."""
    n = len(ranks)
    out = {f"hit@{k}": sum(1 for r in ranks if r and r <= k) / n for k in KS}
    out["mrr"] = sum(1 / r for r in ranks if r) / n
    out["listed"] = sum(1 for r in ranks if r) / n
    out["n"] = n
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--git", required=True, help="Full clone of the indexed repo, at the indexed commit")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--warmup", type=int, default=300, help="Main-line changes used only as history")
    parser.add_argument("--dev-until", default="2022-12-31", help="Changes up to here are the dev set")
    parser.add_argument("--max-files", type=int, default=10, help="Largest change to test (indexed files)")
    parser.add_argument("--no-cochange", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()

    state = get_repo_state(args.repo_id)
    indexed = set(state.metadata_store.indexed_files(args.repo_id))
    changes = read_history(args.git, first_parent=True)[::-1]  # oldest first
    history = CoChange()
    ranks: dict[str, list] = defaultdict(list)
    t0 = time.time()
    for i, change in enumerate(changes):
        files = [f for f in dict.fromkeys(change.files) if f in indexed]
        if i >= args.warmup and 2 <= len(files) <= args.max_files and len(change.files) <= 15:
            split = "dev" if change.date <= args.dev_until else "heldout"
            for hidden in files:
                query = [f for f in files if f != hidden]
                resp = analyze_impact_batch(
                    query, args.repo_id, state.graph, state.metadata_store,
                    depth=3, cochange=None if args.no_cochange else history,
                )
                order = ranked(resp)
                r = order.index(hidden) + 1 if hidden in order else None
                ranks[split].append(r)
                ranks[f"{split}.{kind(hidden)}"].append(r)
        history.add(change, keep=indexed)
    results = {name: summarize(rs) for name, rs in sorted(ranks.items())}
    print(f"\nMissing-file eval ({args.repo_id}); rolling main-line history, {time.time() - t0:.0f}s\n")
    for name, m in results.items():
        cells = "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items())
        print(f"  {name:<18} {cells}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
