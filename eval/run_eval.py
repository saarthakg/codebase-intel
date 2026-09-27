#!/usr/bin/env python3
"""Measure codebase-intel against a labeled benchmark (default: eval/requests_bench.yaml).

Drives the real FastAPI app through TestClient — the same /search, /definition
and /impact endpoints users hit — so the numbers reflect actual behavior, not a
unit-level approximation. No LLM calls, no API key needed.

Usage:
  # score an already-ingested repo_id
  python eval/run_eval.py --repo-id requests

  # (re-)ingest first, then score; save results for before/after comparison
  python eval/run_eval.py --repo-id requests --ingest ../requests-demo --out eval/results/baseline.json

Metrics
  search      file_hit@k  — any expected file among the top-k results
              span_hit@k  — a top-k chunk overlaps an expected symbol's line span
              MRR         — reciprocal rank of the first correct file
              span_mrr    — reciprocal rank of the first chunk overlapping an answer span
              avg_lines@5 — mean line count of the top-5 chunks. Bigger chunks
                            overlap more spans for free, so read span_hit next
                            to this rather than on its own.
  definition  file_acc    — /definition returns the right defining file
              line_acc    — ...and the right line (decorators allowed, ±0 otherwise)
  references  recall / precision of /definition's `references` vs. every file
              that actually uses the symbol (defining file excluded both sides)
  impact      direct_recall_high — true direct importers found in high_confidence
              direct_recall_any  — ...found in any bucket
              high_precision     — high_confidence files that truly depend on the
                                   target (transitively, within the query depth)
  graph       precision / recall of the stored import edges (.py files only)
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from fastapi.testclient import TestClient

from app.main import app, get_repo_state

KS = (1, 5, 10)
IMPACT_DEPTH = 3


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start <= b_end and b_start <= a_end


def eval_search(client: TestClient, repo_id: str, cases: list[dict], verbose: bool, mode: str = "hybrid") -> dict:
    file_hits = {k: [] for k in KS}
    span_hits = {k: [] for k in KS}
    rr: list[float] = []
    span_rr: list[float] = []
    top5_lines: list[int] = []
    misses: list[str] = []
    for case in cases:
        resp = client.post("/search", json={"repo_id": repo_id, "query": case["query"], "top_k": max(KS), "mode": mode})
        resp.raise_for_status()
        results = resp.json()["results"]
        expected_files = {e["file"] for e in case["expected"]}

        first_file_rank = next(
            (i for i, r in enumerate(results, 1) if r["file_path"] in expected_files), None
        )
        first_span_rank = next(
            (
                i for i, r in enumerate(results, 1)
                if any(
                    r["file_path"] == e["file"]
                    and _overlaps(r["start_line"], r["end_line"], e["lines"][0], e["lines"][1])
                    for e in case["expected"]
                )
            ),
            None,
        )
        rr.append(1.0 / first_file_rank if first_file_rank else 0.0)
        span_rr.append(1.0 / first_span_rank if first_span_rank else 0.0)
        top5_lines += [r["end_line"] - r["start_line"] + 1 for r in results[:5]]
        for k in KS:
            file_hits[k].append(1.0 if first_file_rank and first_file_rank <= k else 0.0)
            span_hits[k].append(1.0 if first_span_rank and first_span_rank <= k else 0.0)
        if not (first_span_rank and first_span_rank <= 5):
            top = ", ".join(f"{r['file_path']}:{r['start_line']}" for r in results[:3])
            misses.append(f"  span miss@5  {case['query']!r}\n      want {case['expected'][0]['file']}::"
                          f"{case['expected'][0]['symbol']}  got {top}")

    if verbose and misses:
        print("\n".join(misses))
    out = {f"file_hit@{k}": _mean(file_hits[k]) for k in KS}
    out.update({f"span_hit@{k}": _mean(span_hits[k]) for k in KS})
    out["mrr"] = _mean(rr)
    out["span_mrr"] = _mean(span_rr)
    out["avg_lines@5"] = _mean(top5_lines)
    out["n"] = len(cases)
    return out


def eval_definition(client: TestClient, repo_id: str, cases: list[dict], verbose: bool) -> dict:
    file_ok, line_ok = [], []
    for case in cases:
        resp = client.get("/definition", params={"repo_id": repo_id, "symbol": case["symbol"]})
        if resp.status_code != 200:
            file_ok.append(0.0)
            line_ok.append(0.0)
            if verbose:
                print(f"  definition {case['symbol']!r}: HTTP {resp.status_code}")
            continue
        body = resp.json()
        f_ok = body["defining_file"] == case["file"]
        # The benchmark's line is the first decorator (if any); tree-sitter may
        # report the `def` line instead. Accept anything from the decorator to
        # a few lines below it.
        l_ok = f_ok and body.get("start_line") is not None and 0 <= body["start_line"] - case["line"] <= 3
        file_ok.append(1.0 if f_ok else 0.0)
        line_ok.append(1.0 if l_ok else 0.0)
        if verbose and not l_ok:
            print(f"  definition {case['symbol']!r}: want {case['file']}:{case['line']} "
                  f"got {body['defining_file']}:{body.get('start_line')}")
    return {"file_acc": _mean(file_ok), "line_acc": _mean(line_ok), "n": len(cases)}


def eval_references(client: TestClient, repo_id: str, cases: list[dict], verbose: bool) -> dict:
    recalls, precisions = [], []
    for case in cases:
        resp = client.get("/definition", params={"repo_id": repo_id, "symbol": case["symbol"]})
        if resp.status_code != 200:
            recalls.append(0.0)
            precisions.append(0.0)
            continue
        body = resp.json()
        defining = body["defining_file"]
        got = set(body["references"]) - {defining}
        want = set(case["files"]) - {defining}
        hit = got & want
        recalls.append(len(hit) / len(want) if want else 1.0)
        precisions.append(len(hit) / len(got) if got else (1.0 if not want else 0.0))
        if verbose and want - got:
            print(f"  references {case['symbol']!r}: missing {sorted(want - got)}")
    return {"recall": _mean(recalls), "precision": _mean(precisions), "n": len(cases)}


def _transitive_dependents(graph: dict[str, list[str]], target: str, depth: int) -> set[str]:
    reverse: dict[str, set[str]] = {}
    for src, targets in graph.items():
        for t in targets:
            reverse.setdefault(t, set()).add(src)
    seen: set[str] = set()
    frontier = {target}
    for _ in range(depth):
        frontier = {d for f in frontier for d in reverse.get(f, ())} - seen - {target}
        seen |= frontier
    return seen


def eval_impact(client: TestClient, repo_id: str, cases: list[dict], gt_graph: dict, verbose: bool) -> dict:
    recall_high, recall_any, precision_high = [], [], []
    for case in cases:
        resp = client.post(
            "/impact", json={"repo_id": repo_id, "target": case["target"], "depth": IMPACT_DEPTH}
        )
        resp.raise_for_status()
        body = resp.json()
        high = {i["file_path"] for i in body["high_confidence"]}
        anyb = high | {i["file_path"] for i in body["medium_confidence"] + body["related"]}
        want = set(case["direct_dependents"])
        truly_dependent = _transitive_dependents(gt_graph, case["target"], IMPACT_DEPTH)

        recall_high.append(len(high & want) / len(want) if want else 1.0)
        recall_any.append(len(anyb & want) / len(want) if want else 1.0)
        precision_high.append(len(high & truly_dependent) / len(high) if high else (1.0 if not want else 0.0))
        if verbose and want - high:
            print(f"  impact {case['target']}: direct dependents not in high: {sorted(want - high)}")
    return {
        "direct_recall_high": _mean(recall_high),
        "direct_recall_any": _mean(recall_any),
        "high_precision": _mean(precision_high),
        "n": len(cases),
    }


def eval_graph(repo_id: str, gt_graph: dict) -> dict:
    graph = get_repo_state(repo_id).graph.G
    got = {(s, t) for s, t in graph.edges if s.endswith(".py") and t.endswith(".py")}
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
    parser.add_argument("--ingest", metavar="REPO_PATH", help="(Re-)ingest this path under --repo-id first")
    parser.add_argument("--out", help="Write results JSON here (e.g. eval/results/baseline.json)")
    parser.add_argument("--search-mode", default="hybrid", choices=["hybrid", "semantic", "keyword"])
    parser.add_argument("-v", "--verbose", action="store_true", help="Print individual misses")
    args = parser.parse_args()

    with open(args.bench) as f:
        bench = yaml.safe_load(f)

    client = TestClient(app)
    if args.ingest:
        resp = client.post("/ingest", json={"repo_path": args.ingest, "repo_id": args.repo_id})
        resp.raise_for_status()
        print(f"Ingested: {resp.json()}")

    gt_graph = bench["import_graph"]
    results = {
        "search": eval_search(client, args.repo_id, bench["search"], args.verbose, args.search_mode),
        # Never tune on these two: holdout checks that search gains transfer,
        # identifier covers queries typed as code.
        **{
            name: eval_search(client, args.repo_id, bench[name], args.verbose, args.search_mode)
            for name in ("search_holdout", "search_identifier") if bench.get(name)
        },
        "definition": eval_definition(client, args.repo_id, bench["definition"], args.verbose),
        "references": eval_references(client, args.repo_id, bench["references"], args.verbose),
        "impact": eval_impact(client, args.repo_id, bench["impact"], gt_graph, args.verbose),
        "graph": eval_graph(args.repo_id, gt_graph),
    }

    print(f"\ncodebase-intel eval — {bench['repo']} @ {str(bench.get('repo_commit'))[:10]}\n")
    for section, metrics in results.items():
        cells = []
        for key, val in metrics.items():
            cells.append(f"{key}={val:.3f}" if isinstance(val, float) else f"{key}={val}")
        print(f"  {section:<17} " + "  ".join(cells))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
