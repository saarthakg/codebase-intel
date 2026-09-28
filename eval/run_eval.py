#!/usr/bin/env python3
"""Check the static analysis that impact is built on against a labeled
benchmark (default: eval/requests_bench.yaml).

Impact's structural evidence is only as good as the index under it: which
file defines a changed symbol, who uses it, and which files import which.
These are checked against hand-verified labels.

Usage:
  # score an already-indexed repo_id
  python eval/run_eval.py --repo-id requests

  # (re-)index first, then score
  python eval/run_eval.py --repo-id requests --ingest ../requests-demo --out eval/results/requests.json

Metrics
  definition  file_acc / line_acc — a symbol resolves to the right defining file
              (and line; decorators allowed)
  references  recall / precision of the files that use a symbol (defining file
              excluded both sides)
  impact      direct_recall_any — true direct importers are all listed
  graph       precision / recall of the stored import edges (.py files only)
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from codebase_intel.core.definitions import lookup_definition
from codebase_intel.core.impact import analyze_impact
from codebase_intel.core.pipeline import run_ingestion
from codebase_intel.state import forget_repo, get_repo_state

IMPACT_DEPTH = 3


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def eval_definition(state, repo_id: str, cases: list[dict], verbose: bool) -> dict:
    file_ok, line_ok = [], []
    for case in cases:
        found = lookup_definition(case["symbol"], state.metadata_store, repo_id, state.graph)
        if found is None:
            file_ok.append(0.0)
            line_ok.append(0.0)
            if verbose:
                print(f"  definition {case['symbol']!r}: not found")
            continue
        f_ok = found.defining_file == case["file"]
        # The benchmark's line is the first decorator (if any); tree-sitter may
        # report the `def` line instead. Accept a few lines below it.
        l_ok = f_ok and found.start_line is not None and 0 <= found.start_line - case["line"] <= 3
        file_ok.append(1.0 if f_ok else 0.0)
        line_ok.append(1.0 if l_ok else 0.0)
        if verbose and not l_ok:
            print(f"  definition {case['symbol']!r}: want {case['file']}:{case['line']} "
                  f"got {found.defining_file}:{found.start_line}")
    return {"file_acc": _mean(file_ok), "line_acc": _mean(line_ok), "n": len(cases)}


def eval_references(state, repo_id: str, cases: list[dict], verbose: bool) -> dict:
    recalls, precisions = [], []
    for case in cases:
        found = lookup_definition(case["symbol"], state.metadata_store, repo_id, state.graph)
        if found is None:
            recalls.append(0.0)
            precisions.append(0.0)
            continue
        got = set(found.references) - {found.defining_file}
        want = set(case["files"]) - {found.defining_file}
        hit = got & want
        recalls.append(len(hit) / len(want) if want else 1.0)
        precisions.append(len(hit) / len(got) if got else (1.0 if not want else 0.0))
        if verbose and want - got:
            print(f"  references {case['symbol']!r}: missing {sorted(want - got)}")
    return {"recall": _mean(recalls), "precision": _mean(precisions), "n": len(cases)}


def eval_impact(state, repo_id: str, cases: list[dict], verbose: bool) -> dict:
    recall_any = []
    for case in cases:
        resp = analyze_impact(case["target"], repo_id, state.graph, state.metadata_store,
                              depth=IMPACT_DEPTH, cochange=state.cochange)
        listed = {i.file_path for i in resp.high_confidence + resp.medium_confidence + resp.related}
        want = set(case["direct_dependents"])
        recall_any.append(len(listed & want) / len(want) if want else 1.0)
        if verbose and want - listed:
            print(f"  impact {case['target']}: direct dependents not listed: {sorted(want - listed)}")
    return {"direct_recall_any": _mean(recall_any), "n": len(cases)}


def eval_graph(state, gt_graph: dict) -> dict:
    got = {(s, t) for s, t in state.graph.G.edges if s.endswith(".py") and t.endswith(".py")}
    want = {(s, t) for s, targets in gt_graph.items() for t in targets}
    hit = got & want
    return {
        "edge_recall": len(hit) / len(want) if want else 1.0,
        "edge_precision": len(hit) / len(got) if got else 0.0,
        "edges_found": len(got),
        "edges_true": len(want),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default="requests")
    parser.add_argument("--bench", default=str(Path(__file__).parent / "requests_bench.yaml"))
    parser.add_argument("--ingest", metavar="REPO_PATH", help="(Re-)index this path under --repo-id first")
    parser.add_argument("--out", help="Write results JSON here")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print individual misses")
    args = parser.parse_args()

    with open(args.bench) as f:
        bench = yaml.safe_load(f)
    if args.ingest:
        print(f"Indexed: {run_ingestion(args.ingest, args.repo_id)}")
        forget_repo(args.repo_id)
    state = get_repo_state(args.repo_id)

    results = {
        "definition": eval_definition(state, args.repo_id, bench["definition"], args.verbose),
        "references": eval_references(state, args.repo_id, bench["references"], args.verbose),
        "impact": eval_impact(state, args.repo_id, bench["impact"], args.verbose),
        "graph": eval_graph(state, bench["import_graph"]),
    }

    print(f"\ncodebase-intel static-analysis eval — {bench['repo']} @ {str(bench.get('repo_commit'))[:10]}\n")
    for section, metrics in results.items():
        cells = [f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]
        print(f"  {section:<11} " + "  ".join(cells))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
