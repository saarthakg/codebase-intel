#!/usr/bin/env python3
"""Score "which files use this symbol?" against calls observed at runtime.

Ground truth comes from eval/call_tracer.py: the repo's own test suite run
with a profile hook recording which files called each function. For every
method/function called from at least one *other* file, compare those caller
files with codebase-intel's symbol_users() (the list /impact/diff ranks first).

- recall: observed caller files the tool finds. Trustworthy: every observed
  caller is real.
- precision: tool-claimed files that were observed calling it. A lower bound:
  a claimed caller may be real but untested.

  python eval/run_usage_eval.py --repo-id requests --calls eval/requests_calls.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from codebase_intel.core.usages import symbol_users
from codebase_intel.state import get_repo_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--calls", required=True, help="call_tracer.py output")
    parser.add_argument("--out", help="Write results JSON here")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show the worst cases")
    args = parser.parse_args()

    state = get_repo_state(args.repo_id)
    calls = json.loads(Path(args.calls).read_text())
    groups: dict[str, list[tuple[str, float, float, set, set]]] = {
        "methods": [], "dunder_methods": [], "functions": []}

    for key, callers in calls.items():
        defining_file, qualified = key.split("::", 1)
        truth = set(callers) - {defining_file}
        if not truth or defining_file not in state.graph.G.nodes:
            continue
        if not state.metadata_store.find_symbol(args.repo_id, qualified):
            continue  # not a symbol the index knows (e.g. generated)
        found = set(symbol_users(args.repo_id, qualified, defining_file, state.graph, state.metadata_store))
        found.discard(defining_file)
        recall = len(found & truth) / len(truth)
        precision = len(found & truth) / len(found) if found else 0.0
        method = qualified.rsplit(".", 1)[-1]
        if "." not in qualified:
            group = "functions"
        elif method.startswith("__") and method.endswith("__"):
            group = "dunder_methods"  # mostly invoked implicitly: X(...), with, for, pickle
        else:
            group = "methods"
        groups[group].append((qualified, recall, precision, found - truth, truth - found))

    results = {}
    for group, rows in groups.items():
        if not rows:
            continue
        n = len(rows)
        results[group] = {
            "recall": sum(r[1] for r in rows) / n,
            "precision_lower_bound": sum(r[2] for r in rows) / n,
            "claimed_files_per_symbol": sum(len(r[3]) + len(r[1:2]) for r in rows) / n,
            "n": n,
        }
        if args.verbose:
            worst = sorted(rows, key=lambda r: (r[2], r[1]))[:8]
            for q, rec, prec, extra, missed in worst:
                print(f"  {group.rstrip('s')} {q:<45} recall={rec:.2f} precision>={prec:.2f} "
                      f"extra={sorted(extra)[:3]} missed={sorted(missed)[:3]}")

    print(f"\nSymbol-usage eval ({args.repo_id}) vs runtime calls from the test suite")
    for group, m in results.items():
        print(f"  {group:<15} recall={m['recall']:.3f}  precision>={m['precision_lower_bound']:.3f}  n={m['n']}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
