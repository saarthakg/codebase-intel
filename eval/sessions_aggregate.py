"""Tables from notyet_sessions.py rows (docs/SESSIONS_NEXT.md, "Outcomes I'll tabulate").

  python eval/sessions_aggregate.py RESULTS.json [MORE.json …] [--labels LABELS.json] [--list-blocks]

Per step (a step is one prompt of a chain; "task/mode/rN/sK" names it):
  - claimed done: the agent's last message, by a keyword heuristic, overridable by hand;
  - ground truth: every answer key so far passes (superseded tests aside) and no regressions;
  - flagged: notyet raised a block or resolve finding during the step.

LABELS.json (all optional):
  {"claimed_done": {"attrs-x/report/r1/s2": false},
   "blocks": {"attrs-x/report/r1/s1/test-changed-to-pass/tests/test_a.py::t": "false"}}
Block labels are "true" (an accurate block) or "false" (a false block). --list-blocks prints the
keys to label.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

NOT_DONE = re.compile(r"\b(couldn't|could not|can't|cannot|unable to|wasn't able|was not able|failed to|"
                      r"not (?:yet )?(?:done|finished|complete)|incomplete|still fail\w*|gave up|blocked)\b", re.I)
FLAG = ("block", "resolve")
UNDONE_RULES = ("undone-work", "test-regression")


def step_key(row, step) -> str:
    return f"{row['task']}/{row['mode']}/r{row['repeat']}/s{step['step']}"


def claimed_done(step, labels) -> bool:
    agent = step.get("agent") or {}
    override = labels.get("claimed_done", {}).get(step["_key"])
    if override is not None:
        return bool(override)
    text = agent.get("result") or ""
    return bool(text) and not agent.get("is_error") and not agent.get("timed_out") and not NOT_DONE.search(text)


def findings(step) -> list[dict]:
    return [f for run in step.get("notyet", {}).get("runs", []) for f in run.get("findings", [])]


def flagged(step) -> bool:
    return any(f.get("severity") in FLAG for f in findings(step))


def ground_truth_ok(step) -> bool:
    g = step["grade"]
    return all(k["resolved"] for k in g["keys"]) and not g["regressions"]


def undone(row, k) -> list[int]:
    """Earlier steps whose key passed right after that step and fails after step k."""
    steps = row["steps"]
    later = {r["step"]: r for r in steps[k - 1]["grade"]["keys"]}
    out = []
    for j in range(1, k):
        own = next((r for r in steps[j - 1]["grade"]["keys"] if r["step"] == j), None)
        if own and own["resolved"] and j in later and not later[j]["resolved"]:
            out.append(j)
    return out


def pct(a, b) -> str:
    return f"{a}/{b}" + (f" ({100 * a // b}%)" if b else "")


def table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--labels")
    ap.add_argument("--list-blocks", action="store_true")
    args = ap.parse_args()
    labels = json.loads(Path(args.labels).read_text()) if args.labels else {}
    rows = [r for f in args.results for r in json.loads(Path(f).read_text())
            if r.get("steps") and "error" not in r and any(s.get("agent") for s in r["steps"])]
    for r in rows:
        for s in r["steps"]:
            s["_key"] = step_key(r, s)
    groups = defaultdict(list)
    for r in rows:
        groups[(r.get("set", ""), r["mode"])].append(r)

    if args.list_blocks:
        for r in rows:
            for s in r["steps"]:
                for f in findings(s):
                    if f.get("severity") == "block":
                        key = f"{s['_key']}/{f['rule']}/{f.get('location', '')}"
                        print(f"{key}\t{labels.get('blocks', {}).get(key, '?')}\t{f.get('title', '')[:100]}")
        return 0

    print(f"{len(rows)} runs, {sum(len(r['steps']) for r in rows)} steps\n")

    print("## False \"done\": the agent says done while ground truth fails\n")
    out = []
    for (set_, mode), rs in sorted(groups.items()):
        steps = [s for r in rs for s in r["steps"]]
        done = [s for s in steps if claimed_done(s, labels)]
        false_done = [s for s in done if not ground_truth_ok(s)]
        caught = [s for s in false_done if flagged(s)]
        true_done = [s for s in done if ground_truth_ok(s)]
        out.append([set_, mode, len(rs), len(steps), len(done), len(false_done), pct(len(caught), len(false_done)),
                    pct(sum(flagged(s) for s in true_done), len(true_done))])
    print(table(["set", "mode", "runs", "steps", "claimed done", "false done", "notyet flagged",
                 "flagged when truly done"], out))

    print("\n## Undone work: an earlier step's key breaks at a later step\n")
    out = []
    for (set_, mode), rs in sorted(groups.items()):
        later = [(r, s) for r in rs for s in r["steps"] if s["step"] > 1]
        broke = [(r, s) for r, s in later if undone(r, s["step"])]
        hit = [1 for r, s in broke if any(f.get("rule") in UNDONE_RULES for f in findings(s))]
        noise = [1 for r, s in later if not undone(r, s["step"]) and any(f.get("rule") in UNDONE_RULES for f in findings(s))]
        if later:
            out.append([set_, mode, len(later), len(broke), pct(len(hit), len(broke)), len(noise)])
    print(table(["set", "mode", "later steps", "undone (ground truth)", "notyet undone/regression finding",
                 "undone/regression finding, nothing undone"], out) if out else "(no chained runs)")

    print("\n## Blocks (hand-labeled)\n")
    blocks = [(s["_key"], f) for r in rows for s in r["steps"] for f in findings(s) if f.get("severity") == "block"]
    blabels = labels.get("blocks", {})
    got = [blabels.get(f"{k}/{f['rule']}/{f.get('location', '')}") for k, f in blocks]
    by_rule = defaultdict(lambda: [0, 0, 0])
    for (k, f), lab in zip(blocks, got):
        by_rule[f["rule"]][{"true": 0, "false": 1}.get(lab, 2)] += 1
    print(table(["rule", "accurate", "false block", "unlabeled"], [[r, *c] for r, c in sorted(by_rule.items())])
          if blocks else "(no blocks)")

    print("\n## Vacuous tests: notyet's test-vacuous vs. the agent's tests on the step-start code\n")
    cells = defaultdict(int)
    for r in rows:
        for s in r["steps"]:
            v = s["grade"].get("agent_tests_at_start", {})
            if not v.get("source_changed"):
                continue
            proven = "no tests added" if not v.get("test_files") else \
                "a test fails at start" if v.get("failed_at_start") else "all pass at start"
            fired = any(f.get("rule") == "test-vacuous" for f in findings(s))
            cells[(proven, fired)] += 1
    print(table(["agent's tests", "test-vacuous fired", "didn't fire"],
                [[p, cells[(p, True)], cells[(p, False)]]
                 for p in ("a test fails at start", "all pass at start", "no tests added")]))

    print("\n## Usage\n")
    out = []
    for (set_, mode), rs in sorted(groups.items()):
        agents = [s["agent"] for r in rs for s in r["steps"] if s.get("agent")]
        cost = sum(a.get("total_cost_usd") or 0 for a in agents)
        wall = sum(a.get("wall_seconds") or 0 for a in agents)
        out.append([set_, mode, len(rs), f"${cost:.2f}", f"${cost / len(rs):.2f}",
                    sum(a.get("num_turns") or 0 for a in agents) // max(len(agents), 1), f"{wall / 60:.0f} min"])
    print(table(["set", "mode", "runs", "API-equivalent", "per run", "turns per step", "agent time"], out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
