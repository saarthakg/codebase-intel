"""codebase-intel command line.

  codebase-intel check                  # uncommitted changes in the current repo
  codebase-intel check --base origin/main   # everything on this branch
  codebase-intel check --staged --fail-above 0.8   # pre-commit hook
  codebase-intel impact src/pkg/adapters.py     # one file or symbol
  codebase-intel mcp                    # MCP server over stdio
"""
import argparse
import json
import sys

from app.core.check import CheckResult, check_change
from app.core.workspace import NotAGitRepo


def _progress(msg: str) -> None:
    print(msg, file=sys.stderr)


def format_check(r: CheckResult) -> str:
    lines = [f"Change vs {r.base}: {len(r.changed_files)} file(s)"]
    by_file: dict[str, list[str]] = {}
    for s in r.changed_symbols:
        by_file.setdefault(s.file, []).append(s.symbol)
    for f in r.changed_files:
        syms = by_file.get(f)
        lines.append(f"  {f}" + (f"  ({', '.join(syms)})" if syms else ""))
    if not r.changed_files:
        return "No changes found."

    lines.append("")
    if r.likely_missing:
        lines.append("Likely also needs changing:")
        width = max(len(s.file) for s in r.likely_missing)
        for s in r.likely_missing:
            lines.append(f"  {s.confidence:.2f}  {s.file:<{width}}  {'; '.join(s.reasons)}")
    else:
        lines.append("Nothing else is likely to need changing.")

    callers = [s for s in r.changed_symbols if s.callers_outside_change]
    if callers:
        lines += ["", "Callers of changed code outside this change:"]
        for s in callers:
            shown = ", ".join(s.callers_outside_change[:8])
            more = len(s.callers_outside_change) - 8
            lines.append(f"  {s.symbol}: {shown}" + (f" (+{more} more)" if more > 0 else ""))
    if r.tests_to_run:
        lines += ["", "Tests to run:"] + [f"  {t}" for t in r.tests_to_run]
    if r.not_in_index:
        lines += ["", "Not analyzed (new, or not a source file): " + ", ".join(r.not_in_index)]
    return "\n".join(lines)


def cmd_check(args) -> int:
    result = check_change(args.path, base=args.base, staged=args.staged, min_confidence=args.min_confidence,
                          limit=args.limit, progress=_progress)
    print(result.model_dump_json(indent=2) if args.json else format_check(result))
    if args.fail_above is not None and any(s.confidence >= args.fail_above for s in result.likely_missing):
        return 1
    return 0


def cmd_impact(args) -> int:
    from app.core.impact import analyze_impact
    from app.core.workspace import ensure_index
    _, repo_id, state = ensure_index(args.path, _progress)
    if args.target not in state.graph.G.nodes and not state.metadata_store.find_symbol(repo_id, args.target):
        print(f"'{args.target}' isn't a file (path from the repo root) or a symbol in this repo.", file=sys.stderr)
        return 2
    resp = analyze_impact(args.target, repo_id, state.graph, state.metadata_store, cochange=state.cochange)
    ranked = (resp.high_confidence + resp.medium_confidence + resp.related)[: args.limit]
    if args.json:
        print(json.dumps([{"file": f.file_path, "confidence": round(f.confidence, 2), "reasons": f.reason.split("; ")}
                          for f in ranked], indent=2))
    else:
        width = max((len(f.file_path) for f in ranked), default=0)
        for f in ranked:
            print(f"  {f.confidence:.2f}  {f.file_path:<{width}}  {f.reason}")
    return 0


def cmd_index(args) -> int:
    from app.core.workspace import ensure_index
    root, repo_id, state = ensure_index(args.path, _progress)
    print(f"{root}: {state.graph.G.number_of_nodes()} files indexed ({repo_id})")
    return 0


def cmd_mcp(args) -> int:
    from app.mcp_server import main as mcp_main
    mcp_main()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="codebase-intel", description="Change-impact analysis for git repos.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", help="What else a change probably needs, which tests to run, who calls it")
    p.add_argument("path", nargs="?", default=".", help="Anywhere inside the repo (default: current directory)")
    p.add_argument("--base", help="Compare against the merge base with this ref (e.g. origin/main)")
    p.add_argument("--staged", action="store_true", help="Only staged changes (for a pre-commit hook)")
    p.add_argument("--min-confidence", type=float, default=0.4, help="Hide suggestions below this (default 0.4)")
    p.add_argument("--limit", type=int, default=15, help="At most this many suggestions (default 15)")
    p.add_argument("--fail-above", type=float, metavar="CONFIDENCE",
                   help="Exit 1 if any suggestion reaches this confidence (for hooks and CI)")
    p.add_argument("--json", action="store_true", help="Machine-readable output")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("impact", help="Files likely affected by changing one file or symbol")
    p.add_argument("target", help="File path from the repo root, or a symbol such as HTTPAdapter.send")
    p.add_argument("--path", default=".", help="Anywhere inside the repo (default: current directory)")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_impact)

    p = sub.add_parser("index", help="Index the repo now (otherwise done on first use and when HEAD moves)")
    p.add_argument("path", nargs="?", default=".")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("mcp", help="Run the MCP server over stdio")
    p.set_defaults(func=cmd_mcp)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except NotAGitRepo as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
