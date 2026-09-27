#!/usr/bin/env python3
"""
CLI query tool for codebase-intel.

Usage:
  python scripts/demo_query.py --repo-id <id> "query string"
  python scripts/demo_query.py --repo-id <id> --mode definition --symbol <name>
  python scripts/demo_query.py --repo-id <id> --mode impact --target <file_or_symbol>
  python scripts/demo_query.py --repo-id <id> --mode impact-batch --targets <f1,f2,...>
  git diff HEAD | python scripts/demo_query.py --repo-id <id> --mode impact-diff --diff -
  python scripts/demo_query.py --repo-id <id> --mode ask "question"

Modes: search (default), definition, impact, impact-batch, impact-diff, ask
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.core.search import search_chunks


def load_state(repo_id: str):
    """(faiss_store, metadata_store, graph), loaded exactly as the API server does."""
    from app.main import get_repo_state
    try:
        state = get_repo_state(repo_id)
    except FileNotFoundError:
        print(f"Error: No index found for repo '{repo_id}'. Run ingest_repo.py first.")
        sys.exit(1)
    return state.faiss_store, state.metadata_store, state.graph


def mode_search(repo_id: str, query: str, top_k: int = 10):
    faiss_store, metadata_store, _ = load_state(repo_id)
    results = search_chunks(query, repo_id, top_k, faiss_store, metadata_store)
    print(f"\nSearch: \"{query}\"")
    print(f"Top {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r.file_path}  lines {r.start_line}–{r.end_line}  score={r.score:.3f}")
        print(f"    {r.snippet[:200].strip()}")
        print()


def mode_definition(repo_id: str, symbol: str):
    _, metadata_store, _ = load_state(repo_id)
    from app.core.definitions import lookup_definition
    result = lookup_definition(symbol, metadata_store, repo_id)
    if result is None:
        print(f"Symbol '{symbol}' not found in repo '{repo_id}'.")
        return
    print(f"\nDefinition: {symbol}")
    print(f"  {result.qualified_name} ({result.kind})  "
          f"{result.defining_file}  lines {result.start_line}–{result.end_line}")
    if result.other_definitions:
        print(f"  Other matches ({len(result.other_definitions)}):")
        for d in result.other_definitions[:5]:
            print(f"    - {d.qualified_name} ({d.kind})  {d.file_path}:{d.start_line}")
    if result.references:
        print(f"  Used in {len(result.references)} files ({len(result.reference_locations)} places):")
        for ref in result.references[:10]:
            lines = [str(r.line) for r in result.reference_locations if r.file_path == ref]
            print(f"    - {ref}: {', '.join(lines[:8])}{' …' if len(lines) > 8 else ''}")
    else:
        print("  No references found.")


def _format_impact_line(f) -> str:
    line = f"  [{f.confidence:.2f}] {f.file_path}  — {f.reason}"
    triggered_by = getattr(f, "triggered_by", None)
    if triggered_by:
        line += f"  (via {', '.join(triggered_by)})"
    return line


def _print_impact_buckets(response):
    if response.high_confidence:
        print("HIGH CONFIDENCE:")
        for f in response.high_confidence:
            print(_format_impact_line(f))
    if response.medium_confidence:
        print("\nMEDIUM CONFIDENCE:")
        for f in response.medium_confidence:
            print(_format_impact_line(f))
    if response.related:
        print("\nRELATED:")
        for f in response.related:
            print(_format_impact_line(f))
    if response.tests:
        print("\nTESTS TO RUN:")
        for f in response.tests:
            print(f"  {f.file_path}")


def mode_impact(repo_id: str, target: str, depth: int = 3):
    faiss_store, metadata_store, graph = load_state(repo_id)
    from app.core.impact import analyze_impact
    import app.core.embeddings as embeddings_module
    response = analyze_impact(
        target=target,
        repo_id=repo_id,
        graph=graph,
        faiss_store=faiss_store,
        metadata_store=metadata_store,
        embeddings_module=embeddings_module,
        depth=depth,
        cochange=metadata_store.load_cochange(repo_id),
    )
    print(f"\nImpact analysis: {target}\n")
    _print_impact_buckets(response)


def mode_impact_batch(repo_id: str, targets: list[str], depth: int = 3):
    faiss_store, metadata_store, graph = load_state(repo_id)
    from app.core.impact import analyze_impact_batch
    import app.core.embeddings as embeddings_module
    response = analyze_impact_batch(
        targets=targets,
        repo_id=repo_id,
        graph=graph,
        faiss_store=faiss_store,
        metadata_store=metadata_store,
        embeddings_module=embeddings_module,
        depth=depth,
        cochange=metadata_store.load_cochange(repo_id),
    )
    print(f"\nBatch impact analysis: {', '.join(targets)}\n")
    _print_impact_buckets(response)


def mode_impact_diff(repo_id: str, diff_path: str, depth: int = 3):
    faiss_store, metadata_store, graph = load_state(repo_id)
    from app.core.diff_impact import analyze_diff
    import app.core.embeddings as embeddings_module
    diff = sys.stdin.read() if diff_path == "-" else Path(diff_path).read_text()
    response = analyze_diff(
        diff, repo_id, graph, faiss_store, metadata_store, embeddings_module,
        depth=depth, cochange=metadata_store.load_cochange(repo_id),
    )
    print(f"\nDiff impact: {', '.join(response.targets) or '(no indexed files changed)'}\n")
    if response.changed_symbols:
        print("CHANGED SYMBOLS:")
        for s in response.changed_symbols:
            used = f"  → used in {', '.join(s.used_in)}" if s.used_in else ""
            print(f"  {s.file_path}::{s.qualified_name}{used}")
        print()
    _print_impact_buckets(response)
    if response.unindexed_files:
        print(f"\nNot in the index (skipped): {', '.join(response.unindexed_files)}")


def mode_ask(repo_id: str, question: str, top_k: int = 8, use_cache: bool = True, stream: bool = False):
    faiss_store, metadata_store, _ = load_state(repo_id)
    from app.core.answer import LLMCallError, LLMConfigError, answer_question, stream_answer_question
    from app.models.schemas import AskResponse
    print(f"\nQ: {question}\n")
    try:
        if stream:
            response = None
            for event in stream_answer_question(question, repo_id, faiss_store, metadata_store, top_k, use_cache):
                if event["type"] == "context":
                    # printed only now: retrieval (and model loading, which logs) happens first
                    print("A: ", end="", flush=True)
                elif event["type"] == "delta":
                    print(event["text"], end="", flush=True)
                elif event["type"] == "answer":
                    response = AskResponse(**event["response"])
                    if response.cached:
                        print(response.answer, end="")
            print("\n")
        else:
            response = answer_question(question, repo_id, faiss_store, metadata_store, top_k, use_cache)
            print(f"A: {response.answer}\n")
    except (LLMConfigError, LLMCallError) as e:
        print(f"\nError: {e}")
        sys.exit(1)
    source = "cache" if response.cached else f"{response.backend}/{response.model}"
    if response.citations:
        print("Citations:")
        for c in response.citations:
            print(f"  {c.file_path}  lines {c.start_line}–{c.end_line}  ({c.relevance})")
    if response.uncertainty:
        print(f"\nNote: {response.uncertainty}")
    print(f"\n[{source}; {response.excerpts_used} excerpts, {response.context_chars} chars"
          f"{f', {response.excerpts_omitted} dropped for budget' if response.excerpts_omitted else ''}]")


def main():
    parser = argparse.ArgumentParser(description="Query a codebase-intel index.")
    parser.add_argument("--repo-id", required=True, help="Repository identifier")
    parser.add_argument(
        "--mode", choices=["search", "definition", "impact", "impact-batch", "impact-diff", "ask"],
        default="search"
    )
    parser.add_argument("--symbol", help="Symbol name (for --mode definition)")
    parser.add_argument("--target", help="File path or symbol (for --mode impact)")
    parser.add_argument("--targets", help="Comma-separated file paths (for --mode impact-batch)")
    parser.add_argument("--diff", help="Unified diff file for --mode impact-diff ('-' for stdin)")
    parser.add_argument("--depth", type=int, default=3, help="Graph traversal depth")
    parser.add_argument("--top-k", type=int, default=None, help="Results/chunks (default 10; 8 for ask)")
    parser.add_argument("--no-cache", action="store_true", help="Ask mode: bypass the answer cache")
    parser.add_argument("--stream", action="store_true", help="Ask mode: print the answer as it's generated")
    parser.add_argument("query", nargs="?", help="Search query or question")
    args = parser.parse_args()

    if args.mode == "search":
        if not args.query:
            parser.error("Provide a query string for search mode")
        mode_search(args.repo_id, args.query, args.top_k or 10)
    elif args.mode == "definition":
        if not args.symbol:
            parser.error("--symbol required for definition mode")
        mode_definition(args.repo_id, args.symbol)
    elif args.mode == "impact":
        if not args.target:
            parser.error("--target required for impact mode")
        mode_impact(args.repo_id, args.target, args.depth)
    elif args.mode == "impact-batch":
        if not args.targets:
            parser.error("--targets required for impact-batch mode (comma-separated)")
        mode_impact_batch(args.repo_id, [t.strip() for t in args.targets.split(",") if t.strip()], args.depth)
    elif args.mode == "impact-diff":
        if not args.diff:
            parser.error("--diff required for impact-diff mode (e.g. git diff HEAD > d.patch; --diff d.patch)")
        mode_impact_diff(args.repo_id, args.diff, args.depth)
    elif args.mode == "ask":
        if not args.query:
            parser.error("Provide a question for ask mode")
        mode_ask(args.repo_id, args.query, args.top_k or 8, use_cache=not args.no_cache, stream=args.stream)


if __name__ == "__main__":
    main()
