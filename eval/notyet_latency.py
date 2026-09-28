"""Gate latency and false blocks on real repos (MVP week 2 exit check).

For each repo (a scratch clone with its own .venv), for a sample of source
modules:
  clean  append a comment: the gate must never block (any block is a false block);
  fault  replace the first `return <expr>` in the module with `return None`:
         a crude seeded fault, to see how often the selected tests catch it.
Each edit is checked by a fresh session (baseline = the tree before the edit)
and reverted afterwards. The repos are modified in place, so only point this
at throwaway clones.

  python eval/notyet_latency.py OUT.json REPO [REPO ...] [--files N] [--seed S]
"""
import argparse
import ast
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

from notyet import gate, snapshot, store
from notyet.engines.execution import in_test_dir, is_test_module


def source_modules(root: Path) -> list[str]:
    out = subprocess.run(["git", "-C", str(root), "ls-files", "*.py"], capture_output=True, text=True).stdout
    return sorted(p for p in out.split() if not is_test_module(p) and not in_test_dir(p)
                  and not p.startswith(("docs/", "doc/", "examples/", "scripts/")) and Path(p).name != "conftest.py"
                  and "/" in p)


def fault(text: str) -> str | None:
    """`return <expr>` → `return None` at the first function-level return."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and node.value is not None and not (
                isinstance(node.value, ast.Constant) and node.value.value is None):
            lines = text.splitlines(keepends=True)
            line = lines[node.lineno - 1]
            if node.end_lineno != node.lineno:
                continue
            lines[node.lineno - 1] = line[:node.col_offset] + "return None" + line[node.end_col_offset:]
            return "".join(lines)
    return None


def run_one(root: Path, rel: str, kind: str, baseline: str) -> dict:
    path = root / rel
    original = path.read_text()
    edited = original + "\n# notyet latency probe\n" if kind == "clean" else fault(original)
    if edited is None:
        return {"file": rel, "kind": kind, "skipped": "no single-line return to mutate"}
    path.write_text(edited)
    try:
        session = store.Session(session_id=f"lat-{kind}-{rel}", started=time.time(), baseline_tree=baseline,
                                baseline_head=snapshot.head_tree(str(root)), baseline_source="session-start")
        start = time.monotonic()
        decision = gate.check(str(root), session)
        seconds = time.monotonic() - start
    finally:
        path.write_text(original)
    return {"file": rel, "kind": kind, "seconds": round(seconds, 2), "verdict": decision.verdict,
            "blocks": [f.title for f in decision.findings if f.severity == "block"],
            "resolves": [f.title for f in decision.findings if f.severity == "resolve"]}


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 2) if xs else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("repos", nargs="+")
    ap.add_argument("--files", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--budget", type=int, default=60)
    args = ap.parse_args()
    report = {}
    for repo in args.repos:
        root = Path(repo).resolve()
        (root / ".notyet.toml").write_text(
            f'[test]\ncommand = ".venv/bin/python -m pytest"\nbudget_seconds = {args.budget}\n[gate]\nmode = "enforce"\n')
        baseline = snapshot.snapshot(str(root))
        start = time.monotonic()
        from notyet import pyimports
        pyimports.ImportGraph(str(root), baseline)
        cold = time.monotonic() - start
        files = source_modules(root)
        sample = random.Random(args.seed).sample(files, min(args.files, len(files)))
        rows = []
        for rel in sample:
            for kind in ("clean", "fault"):
                row = run_one(root, rel, kind, baseline)
                rows.append(row)
                print(f"{root.name:10} {kind:5} {row.get('seconds', '-'):>6} {row.get('verdict', row.get('skipped')):12} "
                      f"{rel}", file=sys.stderr)
        clean = [r for r in rows if r["kind"] == "clean"]
        faults = [r for r in rows if r["kind"] == "fault" and "seconds" in r]
        secs = [r["seconds"] for r in rows if "seconds" in r]
        report[root.name] = {
            "import_graph_cold_seconds": round(cold, 2),
            "p50": pct(secs, 0.5), "p95": pct(secs, 0.95), "max": max(secs, default=0),
            "mean": round(statistics.mean(secs), 2) if secs else 0,
            "false_blocks": sum(1 for r in clean if r["blocks"]),
            "clean_edits": len(clean),
            "faults": len(faults),
            "faults_blocked": sum(1 for r in faults if r["verdict"] == "blocked"),
            "faults_not_checked": sum(1 for r in faults if r["verdict"] == "not-checked"),
            "rows": rows,
        }
        (root / ".notyet.toml").unlink()
    Path(args.out).write_text(json.dumps(report, indent=1))
    for name, r in report.items():
        print(f"{name:10} p50 {r['p50']}s p95 {r['p95']}s max {r['max']}s | cold graph {r['import_graph_cold_seconds']}s | "
              f"false blocks {r['false_blocks']}/{r['clean_edits']} | seeded faults blocked "
              f"{r['faults_blocked']}/{r['faults']} (not checked {r['faults_not_checked']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
