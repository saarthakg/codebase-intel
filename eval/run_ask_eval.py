#!/usr/bin/env python3
"""Score /ask answers against questions whose answer location is known.

Reuses the labeled search questions: each names the function(s) that answer
it. Uses whatever LLM backend .env selects (run it with Ollama for $0), and
always calls the model fresh (no cache).

Per question:
- context_has_answer: the answering lines were among the excerpts sent
- cites_answer_file / cites_answer_span: a citation points at the right file /
  overlaps the answering symbol's lines
- names_answer: the answer text mentions the answering symbol's name
- flagged: the answer came back with an uncertainty note

And for the answer checks: how often a flagged answer really was wrong
(flag precision) and how many wrong answers got flagged (flag recall), where
"wrong" means it neither cites the answering span nor names the answer.

  python eval/run_ask_eval.py --repo-id requests --bench eval/requests_bench.yaml --set search_holdout
"""
import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.core.answer import _retrieve, generate_answer, llm_settings
from app.main import get_repo_state


def _overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 <= b1 and b0 <= a1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--bench", required=True)
    parser.add_argument("--set", default="search", help="Which labeled question set to use")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--out", help="Write per-question and summary JSON here")
    args = parser.parse_args()

    cases = yaml.safe_load(Path(args.bench).read_text())[args.set]
    if args.limit:
        cases = cases[: args.limit]
    state = get_repo_state(args.repo_id)
    backend, model = llm_settings()
    rows = []
    for i, case in enumerate(cases, 1):
        chunks = _retrieve(case["query"], args.repo_id, state.faiss_store, state.metadata_store, args.top_k)
        t = time.time()
        resp = generate_answer(case["query"], chunks, args.repo_id, state.metadata_store, use_cache=False)
        seconds = time.time() - t
        exp = case["expected"]
        in_context = any(
            c.file_path == e["file"] and _overlaps(c.start_line, c.end_line, *e["lines"])
            for c in chunks for e in exp
        )
        cites_file = any(c.file_path in {e["file"] for e in exp} for c in resp.citations)
        cites_span = any(
            c.file_path == e["file"] and _overlaps(c.start_line, c.end_line, *e["lines"])
            for c in resp.citations for e in exp
        )
        names = any(e["symbol"].rsplit(".", 1)[-1] in resp.answer for e in exp)
        row = {
            "query": case["query"], "context_has_answer": in_context, "cites_answer_file": cites_file,
            "cites_answer_span": cites_span, "names_answer": names, "flagged": bool(resp.uncertainty),
            "n_citations": len(resp.citations), "unverified": resp.unverified_mentions,
            "seconds": round(seconds, 1),
        }
        rows.append(row)
        print(f"[{i}/{len(cases)}] ctx={in_context:d} span={cites_span:d} name={names:d} "
              f"flag={row['flagged']:d} {seconds:4.0f}s  {case['query'][:60]}", flush=True)

    n = len(rows)
    mean = lambda k: sum(1 for r in rows if r[k]) / n
    wrong = [r for r in rows if not (r["cites_answer_span"] or r["names_answer"])]
    flagged = [r for r in rows if r["flagged"]]
    summary = {
        "backend": backend, "model": model, "n": n,
        "context_has_answer": mean("context_has_answer"),
        "cites_answer_file": mean("cites_answer_file"),
        "cites_answer_span": mean("cites_answer_span"),
        "names_answer": mean("names_answer"),
        "flagged_rate": mean("flagged"),
        "answered_well": 1 - len(wrong) / n,
        "flag_precision": (sum(1 for r in flagged if r in wrong) / len(flagged)) if flagged else None,
        "flag_recall": (sum(1 for r in wrong if r["flagged"]) / len(wrong)) if wrong else None,
        "unverified_rate": sum(1 for r in rows if r["unverified"]) / n,
        "median_seconds": sorted(r["seconds"] for r in rows)[n // 2],
    }
    print("\n" + json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
