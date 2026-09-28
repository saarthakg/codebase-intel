"""Replay real commits through the gate: how often does each rule fire on
changes humans wrote and maintainers merged?

Each commit that changed Python source and tests is treated as one agent
session: baseline = its parent, working tree = the commit. Human commits
aren't all correct, but they were reviewed and merged, so a block here is
most likely a false block; a legitimate behavior change that edits a test's
expectation is the known exception (`test-changed-to-pass`, which the agent
would hand to the user with needs-human).

  python eval/notyet_replay.py OUT.json REPO [REPO ...] [--commits N]

The repos are checked out to each commit in turn, so only use throwaway
clones. Each needs a .venv with the project and its test dependencies.
"""
import argparse
import collections
import json
import subprocess
import sys
import time
from pathlib import Path

from notyet import gate, snapshot, store
from notyet.engines.execution import in_test_dir, is_test_module


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout


def candidate_commits(root: Path, limit: int) -> list[str]:
    """Recent first-parent, non-merge commits touching both source and test .py files."""
    out = git(root, "log", "--first-parent", "--no-merges", "--format=@%H", "--name-only", "-n", "400")
    picked, sha, files = [], None, []

    def consider():
        src = [f for f in files if f.endswith(".py") and not is_test_module(f) and not in_test_dir(f)]
        tests = [f for f in files if is_test_module(f)]
        if sha and src and tests and len(files) <= 20:
            picked.append(sha)

    for line in out.splitlines():
        if line.startswith("@"):
            consider()
            sha, files = line[1:], []
        elif line.strip():
            files.append(line.strip())
    consider()
    return picked[:limit]


def replay(root: Path, sha: str, budget: int) -> dict:
    config = root / ".notyet.toml"
    git(root, "checkout", "-q", "--force", f"{sha}^")
    config.write_text(f'[test]\ncommand = ".venv/bin/python -m pytest"\nbudget_seconds = {budget}\n'
                      f'[gate]\nmode = "enforce"\n')
    baseline = snapshot.snapshot(str(root))          # parent + config
    git(root, "checkout", "-q", "--force", sha)      # the untracked config stays
    session = store.Session(session_id=f"replay-{sha[:10]}", started=time.time(), baseline_tree=baseline,
                            baseline_head=snapshot.head_tree(str(root)), baseline_source="session-start")
    start = time.monotonic()
    decision = gate.check(str(root), session)
    seconds = time.monotonic() - start
    config.unlink()
    subject = git(root, "log", "-1", "--format=%s", sha).strip()
    return {"sha": sha[:10], "subject": subject[:100], "seconds": round(seconds, 1), "verdict": decision.verdict,
            "findings": [{"rule": f.rule, "severity": f.severity, "title": f.title[:200]} for f in decision.findings]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("repos", nargs="+")
    ap.add_argument("--commits", type=int, default=15)
    ap.add_argument("--budget", type=int, default=90)
    args = ap.parse_args()
    report = {}
    for repo in args.repos:
        root = Path(repo).resolve()
        start_ref = git(root, "rev-parse", "HEAD").strip()
        rows = []
        try:
            for sha in candidate_commits(root, args.commits):
                row = replay(root, sha, args.budget)
                rows.append(row)
                rules = ", ".join(sorted({f"{f['rule']}({f['severity']})" for f in row["findings"]})) or "-"
                print(f"{root.name:8} {row['sha']} {row['seconds']:>6}s {row['verdict']:11} {rules}  | {row['subject'][:60]}",
                      file=sys.stderr, flush=True)
        finally:
            git(root, "checkout", "-q", "--force", start_ref)
            (root / ".notyet.toml").unlink(missing_ok=True)
        by_rule = collections.Counter((f["rule"], f["severity"]) for r in rows for f in r["findings"])
        commits_by_rule = collections.Counter(rule for r in rows for rule in {(f["rule"], f["severity"]) for f in r["findings"]})
        report[root.name] = {
            "commits": len(rows),
            "verdicts": dict(collections.Counter(r["verdict"] for r in rows)),
            "commits_with_rule": {f"{r}/{s}": n for (r, s), n in sorted(commits_by_rule.items())},
            "findings_by_rule": {f"{r}/{s}": n for (r, s), n in sorted(by_rule.items())},
            "rows": rows,
        }
    Path(args.out).write_text(json.dumps(report, indent=1))
    for name, r in report.items():
        print(f"{name}: {r['commits']} commits, verdicts {r['verdicts']}\n  commits per rule: {r['commits_with_rule']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
