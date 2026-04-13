#!/usr/bin/env python3
"""
CLI query tool for codebase-intel.

Usage:
  python scripts/demo_query.py --repo-id <id> "query string"
  python scripts/demo_query.py --repo-id <id> --mode definition --symbol <name>
  python scripts/demo_query.py --repo-id <id> --mode impact --target <file_or_symbol>
  python scripts/demo_query.py --repo-id <id> --mode ask "question"

Modes: search (default), definition, impact, ask
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.storage.faiss_store import FAISSStore
from app.storage.metadata_store import MetadataStore
from app.core.graph import DependencyGraph
from app.core.search import search_chunks
from app.core.embeddings import get_embedding_dim

DATA_INDEXES = Path(__file__).parent.parent / "data" / "indexes"
DATA_METADATA = Path(__file__).parent.parent / "data" / "metadata"


def load_state(repo_id: str):
    index_path = str(DATA_INDEXES / f"{repo_id}.index")
    db_path = str(DATA_METADATA / f"{repo_id}.db")
    graph_path = str(DATA_METADATA / f"{repo_id}.graph.pkl")

    if not Path(index_path).exists():
        print(f"Error: No index found for repo '{repo_id}'. Run ingest_repo.py first.")
        sys.exit(1)

    faiss_store = FAISSStore(dim=get_embedding_dim())
    faiss_store.load(index_path)
    metadata_store = MetadataStore(db_path)
    graph = DependencyGraph()
    if Path(graph_path).exists():
        graph.load(graph_path)
    return faiss_store, metadata_store, graph


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
    _, metadata_store, graph = load_state(repo_id)
    results = metadata_store.find_symbol(repo_id, symbol)
    if not results:
        print(f"Symbol '{symbol}' not found in repo '{repo_id}'.")
        return
    defining = next((r for r in results if r["kind"] in ("function", "class", "method")), results[0])
    refs = graph.files_referencing_symbol(symbol, metadata_store, repo_id)
    print(f"\nDefinition: {symbol}")
    print(f"  Defined in: {defining['file_path']}  line {defining['start_line']}  ({defining['kind']})")
    if refs:
        print(f"  Referenced in ({len(refs)} files):")
        for ref in refs[:10]:
            print(f"    - {ref}")
    else:
        print("  No references found.")


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
    )
    print(f"\nImpact analysis: {target}\n")
    if response.high_confidence:
        print("HIGH CONFIDENCE (direct/transitive imports):")
        for f in response.high_confidence:
            print(f"  [{f.confidence:.2f}] {f.file_path}  — {f.reason}")
    if response.medium_confidence:
        print("\nMEDIUM CONFIDENCE:")
        for f in response.medium_confidence:
            print(f"  [{f.confidence:.2f}] {f.file_path}  — {f.reason}")
    if response.related:
        print("\nRELATED (semantic similarity):")
        for f in response.related:
            print(f"  [{f.confidence:.2f}] {f.file_path}  — {f.reason}")


def mode_ask(repo_id: str, question: str, top_k: int = 8):
    faiss_store, metadata_store, _ = load_state(repo_id)
    from app.core.answer import generate_answer
    chunks_results = search_chunks(question, repo_id, top_k, faiss_store, metadata_store)
    # Convert SearchResult back to ChunkMetadata for generate_answer
    chunk_metas = [metadata_store.get_chunk(r.chunk_id) for r in chunks_results]
    chunk_metas = [c for c in chunk_metas if c is not None]

    response = generate_answer(question, chunk_metas, repo_id)
    print(f"\nQ: {question}\n")
    print(f"A: {response.answer}\n")
    if response.citations:
        print("Citations:")
        for c in response.citations:
            print(f"  {c.file_path}  lines {c.start_line}–{c.end_line}")
            print(f"    {c.relevance}")
    if response.uncertainty:
        print(f"\nNote: {response.uncertainty}")


def main():
    parser = argparse.ArgumentParser(description="Query a codebase-intel index.")
    parser.add_argument("--repo-id", required=True, help="Repository identifier")
    parser.add_argument("--mode", choices=["search", "definition", "impact", "ask"], default="search")
    parser.add_argument("--symbol", help="Symbol name (for --mode definition)")
    parser.add_argument("--target", help="File path or symbol (for --mode impact)")
    parser.add_argument("--depth", type=int, default=3, help="Graph traversal depth")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("query", nargs="?", help="Search query or question")
    args = parser.parse_args()

    if args.mode == "search":
        if not args.query:
            parser.error("Provide a query string for search mode")
        mode_search(args.repo_id, args.query, args.top_k)
    elif args.mode == "definition":
        if not args.symbol:
            parser.error("--symbol required for definition mode")
        mode_definition(args.repo_id, args.symbol)
    elif args.mode == "impact":
        if not args.target:
            parser.error("--target required for impact mode")
        mode_impact(args.repo_id, args.target, args.depth)
    elif args.mode == "ask":
        if not args.query:
            parser.error("Provide a question for ask mode")
        mode_ask(args.repo_id, args.query, args.top_k)


if __name__ == "__main__":
    main()
