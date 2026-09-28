#!/usr/bin/env python3
"""Fail (exit 1) if any eval result misses its bound in eval/thresholds.yaml.

  python eval/check_thresholds.py requests=out/requests.json replay_requests=out/replay.json ...

Each argument is LABEL=PATH, where LABEL is a section of thresholds.yaml. A
bound is a floor (`metric: 0.5`) or, for lower-is-better metrics, a ceiling
(`metric: {max: 0.8}`). Metric keys are paths into the results JSON, e.g.
`shipped.all.false_alarms@p0.2`.
"""
import argparse
import json
import sys
from pathlib import Path

import yaml


def lookup(results, key: str):
    """results["a"]["b.c"] for key "a.b.c": keys may themselves contain dots."""
    if not isinstance(results, dict):
        return None
    if key in results:
        return results[key]
    parts = key.split(".")
    for i in range(1, len(parts)):
        head = ".".join(parts[:i])
        if head in results:
            return lookup(results[head], ".".join(parts[i:]))
    return None


def check(results: dict, bounds: dict, label: str) -> list[str]:
    failures = []
    for key, bound in bounds.items():
        value = lookup(results, key)
        if isinstance(bound, dict):
            ok, desc = value is not None and value <= bound["max"], f"max {bound['max']}"
        else:
            ok, desc = value is not None and value >= bound, f"floor {bound}"
        shown = f"{value:.3f}" if isinstance(value, float) else str(value)
        print(f"  {'ok  ' if ok else 'FAIL'} {label}.{key:<38} {shown:<8} ({desc})")
        if not ok:
            failures.append(f"{label}.{key}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", help="LABEL=PATH pairs, e.g. requests=out/requests.json")
    parser.add_argument("--thresholds", default=str(Path(__file__).parent / "thresholds.yaml"))
    args = parser.parse_args()

    floors = yaml.safe_load(Path(args.thresholds).read_text())
    failures = []
    for pair in args.results:
        label, _, path = pair.partition("=")
        if label not in floors or not path:
            parser.error(f"{pair!r}: expected LABEL=PATH with LABEL one of {', '.join(floors)}")
        failures += check(json.loads(Path(path).read_text()), floors[label], label)
    if failures:
        print(f"\n{len(failures)} metric(s) out of bounds: {', '.join(failures)}")
        sys.exit(1)
    print("\nAll metrics within their bounds.")


if __name__ == "__main__":
    main()
