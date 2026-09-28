#!/usr/bin/env python3
"""Score replay datasets (eval/build_replay.py) with different scorers.

For hidden-file replays: hit@k and MRR of the hidden file. At confidence
thresholds: recall (the hidden file is flagged), precision (flagged files
that are the hidden one) and, on complete changes, false alarms per change.

  python eval/score_replay.py replay_requests.jsonl.gz [more...] --scorer baseline --scorer logistic
"""
import argparse
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

KS = (1, 3, 5, 10)
THRESHOLDS = (0.1, 0.2, 0.3, 0.5)


def load(path: str) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


# ── Scorers: (candidate dict) -> score. Higher is better. ─────────────────────

def baseline(c: dict) -> float:
    """The shipped noisy-OR (impact.py + diff_impact.py)."""
    ev = []
    if c["import_hops"]:
        ev.append(0.40)
    if c["symbol_uses"]:
        ev.append(0.96)
    if c["named_test"]:
        ev.append(0.97)
    if c["cc_n"] >= 2:
        ev.append(min(0.9, 0.4 + 0.5 * c["cc_p"]))
    miss = 1.0
    for e in ev:
        miss *= 1 - e
    return 1 - miss


def baseline_key(c: dict) -> tuple:
    """Shipped ordering: confidence, then co-change p, then base rate."""
    return (baseline(c), c["cc_p"], c["commits"])


from codebase_intel.core.scoring import FEATURES, featurize  # noqa: E402  (one definition, shared with the product)


def fit_logistic(rows: list[tuple[list[float], int]], l2: float = 1.0, iters: int = 300):
    """Plain logistic regression by Newton's method (numpy only in the eval)."""
    import numpy as np
    X = np.array([[1.0, *x] for x, _ in rows])
    y = np.array([t for _, t in rows], dtype=float)
    mu, sd = X[:, 1:].mean(0), X[:, 1:].std(0) + 1e-9
    X[:, 1:] = (X[:, 1:] - mu) / sd
    w = np.zeros(X.shape[1])
    reg = np.eye(len(w)) * l2
    reg[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + reg @ w
        H = (X * (p * (1 - p))[:, None]).T @ X + reg
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-6:
            break

    def predict(c: dict) -> float:
        x = (np.array(featurize(c)) - mu) / sd
        return float(1 / (1 + np.exp(-(w[0] + x @ w[1:]))))
    predict.weights = dict(zip(["bias", *FEATURES], w / np.r_[1, sd]))
    return predict


# ── Metrics ───────────────────────────────────────────────────────────────────

def evaluate(queries: list[dict], score, key=None) -> dict:
    """`score`: candidate -> score, or a dict query-id -> that function (rolling models)."""
    if isinstance(score, dict):
        per_query = score
        queries = [q for q in queries if id(q) in per_query]
        return _evaluate(queries, lambda q: per_query[id(q)], lambda q: per_query[id(q)])
    return _evaluate(queries, lambda q: score, lambda q: key or score)


def rolling_logistic(queries: list[dict], retrain_every: int = 100, window: Optional[int] = None,
                     min_train: int = 200) -> dict:
    """As in use: each change is scored by a model trained only on earlier
    changes (the last `window` hidden-file replays, or all), refit every
    `retrain_every` changes."""
    ordered = sorted(queries, key=lambda q: q["change"])
    out, model, last_fit = {}, None, None
    for q in ordered:
        if last_fit is None or q["change"] - last_fit >= retrain_every:
            past = [p for p in ordered if p["change"] < q["change"] and p["hidden"]]
            if window:
                past = past[-window:]
            if len(past) >= min_train:
                model = fit_logistic([(featurize(c), int(c["file"] == p["hidden"])) for p in past
                                      for c in p["candidates"]])
                last_fit = q["change"]
        if model is not None:
            out[id(q)] = model
    return out


def _evaluate(queries: list[dict], score_of, key_of) -> dict:
    hidden = [q for q in queries if q["hidden"]]
    complete = [q for q in queries if not q["hidden"]]
    ranks = []
    flagged = {t: [0, 0, 0] for t in THRESHOLDS}  # hits, suggestions, hidden flagged
    for q in hidden:
        score, key = score_of(q), key_of(q)
        order = sorted(q["candidates"], key=key, reverse=True)
        files = [c["file"] for c in order]
        ranks.append(files.index(q["hidden"]) + 1 if q["hidden"] in files else None)
        for t in THRESHOLDS:
            shown = [c for c in q["candidates"] if score(c) >= t]
            hit = any(c["file"] == q["hidden"] for c in shown)
            flagged[t][0] += hit
            flagged[t][1] += len(shown)
    n = len(hidden)
    out = {f"hit@{k}": sum(1 for r in ranks if r and r <= k) / n for k in KS}
    out["mrr"] = sum(1 / r for r in ranks if r) / n
    for t in THRESHOLDS:
        hits, shown, _ = flagged[t]
        alarms = sum(sum(1 for c in q["candidates"] if score_of(q)(c) >= t) for q in complete) / max(len(complete), 1)
        out[f"recall@p{t}"] = hits / n
        out[f"precision@p{t}"] = hits / shown if shown else 0.0
        out[f"false_alarms@p{t}"] = alarms
    out["n"] = n
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--dev-until", default="2022-12-31")
    parser.add_argument("--scorer", action="append", default=[],
                        choices=["shipped", "baseline", "cc", "rc", "logistic", "rolling", "rolling500"])
    parser.add_argument("--out", help="Write results JSON here (scorer -> split -> metrics)")
    parser.add_argument("--weights", action="store_true", help="Print the fitted logistic weights")
    args = parser.parse_args()
    scorers = args.scorer or ["baseline", "logistic"]

    results: dict = {}
    for path in args.datasets:
        queries = load(path)
        dev = [q for q in queries if q["date"] <= args.dev_until]
        held = [q for q in queries if q["date"] > args.dev_until]
        print(f"\n{Path(path).name}: dev {sum(1 for q in dev if q['hidden'])} / held-out "
              f"{sum(1 for q in held if q['hidden'])} hidden-file replays")
        rolling = {}
        if any(n.startswith("rolling") for n in scorers):
            rolling = {n: rolling_logistic(queries, window=500 if n == "rolling500" else None)
                       for n in scorers if n.startswith("rolling")}
            covered = set.intersection(*(set(r) for r in rolling.values()))
            dev = [q for q in dev if id(q) in covered]
            held = [q for q in held if id(q) in covered]
        for name in scorers:
            if name == "shipped":
                from codebase_intel.core.scoring import probability
                score, key = probability, None
            elif name in rolling:
                score, key = rolling[name], None
            elif name == "rc":
                score, key = (lambda c: c.get("rc_p", 0.0) + 0.001 * c["cc_p"]), None
            elif name == "baseline":
                score, key = baseline, baseline_key
            elif name == "cc":
                score, key = (lambda c: min(0.9, 0.4 + 0.5 * c["cc_p"]) if c["cc_n"] >= 2 else 0.0), None
            else:
                train = [(featurize(c), int(c["file"] == q["hidden"])) for q in dev if q["hidden"]
                         for c in q["candidates"]]
                score = fit_logistic(train)
                key = None
                if args.weights:
                    print("   weights: " + ", ".join(f"{k} {v:+.2f}" for k, v in score.weights.items()))
            for split, qs in (("dev", dev), ("heldout", held), ("all", dev + held)):
                if not any(q["hidden"] for q in qs):
                    continue
                m = evaluate(qs, score, key)
                results.setdefault(name, {})[split] = m
                cells = "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items())
                print(f"  {name:<9} {split:<7} {cells}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
