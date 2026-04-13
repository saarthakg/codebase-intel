#!/usr/bin/env python3
"""CLI: python scripts/ingest_repo.py --repo <path> --repo-id <name>"""
import argparse
import os
import sys
from pathlib import Path

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from tqdm import tqdm

from app.core.ingest import walk_repo, detect_language, load_file
from app.core.chunking import chunk_file
from app.core.symbols import extract_symbols, extract_imports
from app.core.graph import DependencyGraph, resolve_python_import, resolve_ts_import
from app.core.embeddings import embed_texts, get_embedding_dim
from app.storage.faiss_store import FAISSStore
from app.storage.metadata_store import MetadataStore


DATA_INDEXES = Path(__file__).parent.parent / "data" / "indexes"
DATA_METADATA = Path(__file__).parent.parent / "data" / "metadata"


def ingest(repo_path: str, repo_id: str) -> dict:
    repo_path = str(Path(repo_path).resolve())
    DATA_INDEXES.mkdir(parents=True, exist_ok=True)
    DATA_METADATA.mkdir(parents=True, exist_ok=True)

    db_path = str(DATA_METADATA / f"{repo_id}.db")
    index_path = str(DATA_INDEXES / f"{repo_id}.index")
    graph_path = str(DATA_METADATA / f"{repo_id}.graph.pkl")

    metadata_store = MetadataStore(db_path)
    graph = DependencyGraph()

    # ── Step 1: Walk repo ──────────────────────────────────────────────────────
    print(f"Walking repo: {repo_path}")
    file_paths = walk_repo(repo_path)
    print(f"  Found {len(file_paths)} files")

    # ── Step 2: Parse, chunk, store symbols ───────────────────────────────────
    all_chunks = []
    total_symbols = 0
    file_contents: dict[str, str] = {}  # cache for import resolution

    for file_path in tqdm(file_paths, desc="Parsing files"):
        content = load_file(file_path)
        if content is None:
            continue
        rel_path = os.path.relpath(file_path, repo_path)
        language = detect_language(file_path)
        file_contents[file_path] = content

        # Extract symbols and store them
        symbols = extract_symbols(content, rel_path, language)
        imports = extract_imports(content, rel_path, language)
        graph.add_file(rel_path)

        for sym in symbols:
            metadata_store.upsert_symbol(
                name=sym.name,
                repo_id=repo_id,
                file_path=rel_path,
                line=sym.start_line,
                kind=sym.kind,
            )
            total_symbols += 1

        # Chunk the file
        chunks = chunk_file(content, rel_path, language)
        for chunk in chunks:
            metadata_store.upsert_chunk(chunk, repo_id)
        all_chunks.extend(chunks)

        # Store edges from imports
        for imp in imports:
            resolved = None
            if language == "python":
                resolved = resolve_python_import(imp.imported_module, file_path, repo_path)
            elif language in ("typescript", "javascript"):
                resolved = resolve_ts_import(imp.imported_module, file_path, repo_path)
            if resolved:
                rel_target = os.path.relpath(resolved, repo_path)
                graph.add_import_edge(rel_path, rel_target)
                metadata_store.upsert_edge(repo_id, rel_path, rel_target, "import")

    # ── Step 3: Save graph ────────────────────────────────────────────────────
    graph.save(graph_path)
    edge_count = graph.edge_count

    # ── Step 4: Embed and index ───────────────────────────────────────────────
    if not all_chunks:
        print("No chunks to embed.")
        return {}

    print(f"Embedding {len(all_chunks)} chunks...")
    texts = [c.content for c in all_chunks]
    chunk_ids = [c.chunk_id for c in all_chunks]

    embeddings = embed_texts(texts)
    dim = embeddings.shape[1]
    faiss_store = FAISSStore(dim=dim)
    faiss_store.add(embeddings, chunk_ids)
    faiss_store.save(index_path)

    summary = {
        "repo_id": repo_id,
        "files_indexed": len(file_paths),
        "chunks_indexed": len(all_chunks),
        "symbols_extracted": total_symbols,
        "edges_in_graph": edge_count,
    }

    print(
        f"\nIndexed {summary['files_indexed']} files, "
        f"{summary['chunks_indexed']} chunks, "
        f"{summary['symbols_extracted']} symbols, "
        f"{summary['edges_in_graph']} graph edges."
    )
    print(f"Saved to {index_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Ingest a code repository into codebase-intel.")
    parser.add_argument("--repo", required=True, help="Path to repository root")
    parser.add_argument("--repo-id", required=True, help="Identifier for this repo (e.g. 'requests')")
    args = parser.parse_args()
    ingest(args.repo, args.repo_id)


if __name__ == "__main__":
    main()
