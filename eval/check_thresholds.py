#!/usr/bin/env python3
"""Fail (exit 1) if any eval result is below its floor in eval/thresholds.yaml.

  python eval/check_thresholds.py --main results/main.json --history results/history.json
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
        status = "ok  " if value is not None and value >= floor else "FAIL"
        print(f"  {status} {label}.{key:<32} {value if value is not None else 'missing'!s:<8} (floor {floor})")
        if status == "FAIL":
            failures.append(f"{label}.{key}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", help="run_eval.py --out JSON")
    parser.add_argument("--history", help="run_history_eval.py --out JSON")
    parser.add_argument("--thresholds", default=str(Path(__file__).parent / "thresholds.yaml"))
    args = parser.parse_args()

    floors = yaml.safe_load(Path(args.thresholds).read_text())
    failures = []
    for label, path in (("main", args.main), ("history", args.history)):
        if path:
            failures += check(json.loads(Path(path).read_text()), floors[label], label)
    if failures:
        print(f"\n{len(failures)} metric(s) below threshold: {', '.join(failures)}")
        sys.exit(1)
    print("\nAll metrics at or above their floors.")


if __name__ == "__main__":
    main()
