#!/usr/bin/env python3
"""Exact-string lookup: does each search mode find a pasted string?

Questions are generated, not picked: string literals of 20-80 characters that
occur exactly once in the repo's non-test source (sampled with a fixed seed).
Each is queried twice, as written and as a user would see it at runtime, with
`{name}` / `%s` placeholders filled in (e.g. a pasted error message). A hit is
a top-k chunk containing the literal's line.

  python eval/run_exact_eval.py --repo-id requests --source <checkout>
"""
import argparse
import ast
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.search import search_chunks
from app.state import get_repo_state

_PLACEHOLDER = re.compile(r"\{[^{}]*\}|%[-#0 +]*\d*(?:\.\d+)?[sdrfix]")
MODES = ("semantic", "hybrid", "keyword")


def literals(source: Path, files: list[str]) -> list[tuple[str, str, int]]:
    """(text, file, line) for each string literal in these files."""
    out = []
    for rel in files:
        try:
            tree = ast.parse((source / rel).read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                out.append((node.value, rel, node.lineno))
            elif isinstance(node, ast.JoinedStr):  # f"..." → "{}" placeholders
                text = "".join(v.value if isinstance(v, ast.Constant) else "{}" for v in node.values)
                out.append((text, rel, node.lineno))
    return out


def questions(source: Path, files: list[str], n: int, seed: int) -> list[dict]:
    lits = literals(source, files)
    counts = Counter(t for t, _, _ in lits)
    pool = sorted({
        (t, f, line) for t, f, line in lits
        if counts[t] == 1 and 20 <= len(t) <= 80 and "\n" not in t and len(t.split()) >= 3
    })
    rng = random.Random(seed)
    picked = rng.sample(pool, min(n, len(pool)))
    return [{"literal": t, "rendered": _PLACEHOLDER.sub("example", t), "file": f, "line": line}
            for t, f, line in picked]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--source", required=True, help="Checkout the repo was indexed from")
    parser.add_argument("-n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out")
    args = parser.parse_args()

    state = get_repo_state(args.repo_id)
    files = sorted({
        f for f in state.metadata_store.indexed_files(args.repo_id)
        if f.endswith(".py") and not re.search(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*$|conftest\.py$", f)
    })
    qs = questions(Path(args.source), files, args.n * 2, args.seed)
    # Keep questions the index agrees with (same code at that line), in case
    # the checkout and the index differ.
    def indexed(q):
        return any(c.start_line <= q["line"] <= c.end_line and q["literal"].split("{")[0][:15] in c.content
                   for c in state.metadata_store.get_chunks_by_file(args.repo_id, q["file"]))
    qs = [q for q in qs if indexed(q)][: args.n]
    results = {}
    for variant in ("literal", "rendered"):
        for mode in MODES:
            hits = 0
            for q in qs:
                found = search_chunks(q[variant], args.repo_id, args.top_k, state.faiss_store,
                                      state.metadata_store, mode=mode)
                hits += any(r.file_path == q["file"] and r.start_line <= q["line"] <= r.end_line for r in found)
            results[f"{variant}.{mode}"] = hits / len(qs)
    print(f"{args.repo_id}: {len(qs)} unique string literals, hit@{args.top_k}")
    for variant in ("literal", "rendered"):
        print(f"  {variant:<9} " + "  ".join(f"{m} {results[f'{variant}.{m}']:.2f}" for m in MODES))
    if args.out:
        Path(args.out).write_text(json.dumps({"n": len(qs), "hit@k": results, "questions": qs}, indent=2))


if __name__ == "__main__":
    main()
