#!/usr/bin/env python3
"""Fail (exit 1) if any eval result is below its floor in eval/thresholds.yaml.

  python eval/check_thresholds.py requests=out/requests.json requests_history=out/history.json ...

Each argument is LABEL=PATH, where LABEL is a section of thresholds.yaml.
"""
import argparse
import json
import sys
from pathlib import Path

import yaml


def check(results: dict, floors: dict, label: str) -> list[str]:
    failures = []
    for key, floor in floors.items():
        section, metric = key.split(".", 1)
        value = results.get(section, {}).get(metric)
        ok = value is not None and value >= floor
        shown = f"{value:.3f}" if isinstance(value, float) else str(value)
        print(f"  {'ok  ' if ok else 'FAIL'} {label}.{key:<32} {shown:<8} (floor {floor})")
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
        print(f"\n{len(failures)} metric(s) below threshold: {', '.join(failures)}")
        sys.exit(1)
    print("\nAll metrics at or above their floors.")


if __name__ == "__main__":
    main()
