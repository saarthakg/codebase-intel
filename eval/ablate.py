#!/usr/bin/env python3
"""Ablations: what is each feature worth today? Turns features off one at a
time and re-runs the relevant eval.

  # impact signals, on the commit-history eval (needs a full clone)
  python eval/ablate.py impact --repo-id requests --git <full clone>

  # search features: re-indexes a scratch copy of the repo per variant
  python eval/ablate.py search --repo-id requests --source ../requests-demo --bench eval/requests_bench.yaml

Nothing here changes the product; toggles are applied in-process.
"""
import argparse
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))


def ablate_impact(repo_id: str, git: str) -> None:
    import app.core.graph as graph_mod
    import app.core.impact as impact
    import run_history_eval

    originals = (impact.tests_named_for, impact._SEMANTIC_FILES, graph_mod.DependencyGraph.dependents_of)

    def run(label, extra=(), no_named_tests=False, no_semantic=False, no_graph=False):
        impact.tests_named_for = (lambda *a, **k: []) if no_named_tests else originals[0]
        impact._SEMANTIC_FILES = 0 if no_semantic else originals[1]
        graph_mod.DependencyGraph.dependents_of = (lambda self, f, depth=3: []) if no_graph else originals[2]
        with tempfile.NamedTemporaryFile(suffix=".json") as out:
            sys.argv = ["x", "--git", git, "--repo-id", repo_id, "--out", out.name, *extra]
            with contextlib.redirect_stdout(io.StringIO()):
                run_history_eval.main()
            r = json.loads(Path(out.name).read_text())
        fmt = lambda s: f"{r[s]['recall@5']:.3f} / {r[s]['recall@10']:.3f} / {r[s]['mrr']:.3f}"
        print(f"  {label:<18} dev {fmt('dev')}   held-out {fmt('heldout')}", flush=True)

    print(f"Impact ablation ({repo_id}), recall@5 / recall@10 / MRR")
    run("full")
    run("- co-change", ["--no-cochange"])
    run("- import graph", no_graph=True)
    run("- named tests", no_named_tests=True)
    run("- semantic", no_semantic=True)
    run("import graph only", ["--no-cochange"], no_named_tests=True, no_semantic=True)
    impact.tests_named_for, impact._SEMANTIC_FILES, graph_mod.DependencyGraph.dependents_of = originals


def ablate_search(repo_id: str, source: str, bench: str) -> None:
    import app.core.pipeline as pipeline
    import run_eval
    from fastapi.testclient import TestClient
    from app.main import app

    orig_header, orig_chunk = pipeline.embedding_text, pipeline.chunk_file

    def run(label, no_header=False, windows=False):
        pipeline.embedding_text = (lambda c, s=None: c.content) if no_header else orig_header
        pipeline.chunk_file = (
            (lambda content, path, lang, symbols=None, imports=None, **kw: orig_chunk(content, path, lang, **kw))
            if windows else orig_chunk
        )
        rid = f"ablate-{repo_id}-{label}"
        with tempfile.NamedTemporaryFile(suffix=".json") as out:
            sys.argv = ["x", "--repo-id", rid, "--bench", bench, "--ingest", source, "--out", out.name]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                run_eval.main()
            r = json.loads(Path(out.name).read_text())
        TestClient(app).delete(f"/repos/{rid}")
        cells = "   ".join(
            f"{k} top5 {v['span_hit@5']:.3f} mrr {v['span_mrr']:.3f}" for k, v in r.items() if k.startswith("search"))
        print(f"  {label:<12} {cells}", flush=True)

    print(f"Search ablation ({repo_id})")
    run("full")
    run("no-header", no_header=True)
    run("windows", windows=True)
    pipeline.embedding_text, pipeline.chunk_file = orig_header, orig_chunk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="what", required=True)
    p_imp = sub.add_parser("impact")
    p_imp.add_argument("--repo-id", required=True)
    p_imp.add_argument("--git", required=True)
    p_s = sub.add_parser("search")
    p_s.add_argument("--repo-id", required=True)
    p_s.add_argument("--source", required=True)
    p_s.add_argument("--bench", required=True)
    args = parser.parse_args()
    if args.what == "impact":
        ablate_impact(args.repo_id, args.git)
    else:
        ablate_search(args.repo_id, args.source, args.bench)


if __name__ == "__main__":
    main()
