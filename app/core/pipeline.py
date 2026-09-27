"""The full ingest pipeline, shared by the HTTP route and the CLI script.

Previously this ~70-line sequence (walk → chunk → extract symbols/imports →
build graph → embed → save) was duplicated almost verbatim in
app/api/routes_ingest.py and scripts/ingest_repo.py. Any fix (e.g. clearing
stale rows before re-ingest, validating repo_id) had to be made twice and was
easy to miss in one of the two places. This module is now the single
implementation; both callers just format the result for their own interface.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from app.core import paths
from app.core.chunking import chunk_file
from app.core.embeddings import embed_texts, get_embedding_dim, get_embedding_model_name
from app.core.graph import (
    DependencyGraph,
    find_python_source_roots,
    load_ts_config,
    resolve_python_import,
    resolve_ts_import,
)
from app.core.ingest import detect_language, load_file, walk_repo
from app.core.symbols import analyze_file
from app.core.validation import validate_repo_id
from app.storage.faiss_store import FAISSStore
from app.storage.metadata_store import MetadataStore

ProgressFn = Optional[Callable[[str], None]]


class IngestError(ValueError):
    """Raised for user-fixable ingest problems (bad path, bad repo_id, etc.)."""


def run_ingestion(repo_path: str, repo_id: str, progress: ProgressFn = None) -> dict:
    """Ingest `repo_path` under `repo_id`. Returns a dict of summary counts.

    Safe to call repeatedly with the same repo_id — each run fully replaces
    the previous one (stale chunks/symbols/edges from an earlier ingest of
    this repo_id are cleared first, and the FAISS index/graph are rebuilt
    from scratch rather than appended to).
    """
    validate_repo_id(repo_id)
    resolved_repo_path = str(Path(repo_path).resolve())
    if not Path(resolved_repo_path).is_dir():
        raise IngestError(f"repo_path is not a directory: {resolved_repo_path}")

    def _report(msg: str) -> None:
        if progress:
            progress(msg)

    paths.ensure_data_dirs()

    metadata_store = MetadataStore(str(paths.db_path(repo_id)))
    backend = os.environ.get("EMBEDDING_BACKEND", "local").lower()
    model_name = get_embedding_model_name(backend)
    try:
        graph, all_chunks, file_count = _index_files(
            resolved_repo_path, repo_id, metadata_store, _report
        )
        if all_chunks:
            _report(f"Embedding {len(all_chunks)} chunks...")
            embeddings = embed_texts([c.content for c in all_chunks], backend=backend, model=model_name)
            faiss_store = FAISSStore(
                dim=embeddings.shape[1], embedding_backend=backend, embedding_model=model_name
            )
            faiss_store.add(embeddings, [c.chunk_id for c in all_chunks])
        else:
            # Still write an (empty) index so a repo with zero indexable chunks
            # doesn't leave get_repo_state() unable to find anything to load.
            faiss_store = FAISSStore(
                dim=get_embedding_dim(backend, model_name), embedding_backend=backend,
                embedding_model=model_name,
            )
        # Commit the DB rebuild only once everything that can fail slowly
        # (parsing, embedding) has succeeded: an error mid-ingest leaves the
        # previous index intact instead of a half-cleared DB that no longer
        # matches the FAISS index on disk.
        metadata_store.commit()
    except BaseException:
        metadata_store.rollback()
        raise
    total_symbols = metadata_store.count_symbols(repo_id)
    total_references = metadata_store.count_references(repo_id)
    metadata_store.close()

    faiss_store.save(str(paths.index_path(repo_id)))
    graph.save(str(paths.graph_path(repo_id)))
    edge_count = graph.edge_count

    summary = {
        "repo_id": repo_id,
        "files_indexed": file_count,
        "chunks_indexed": len(all_chunks),
        "symbols_extracted": total_symbols,
        "references_indexed": total_references,
        "edges_in_graph": edge_count,
        "embedding_backend": backend,
        "embedding_model": model_name,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(paths.meta_path(repo_id), "w") as f:
        json.dump(summary, f)

    return summary


def _index_files(
    repo_path: str, repo_id: str, metadata_store: MetadataStore, report: Callable[[str], None]
) -> tuple[DependencyGraph, list, int]:
    """Walk, parse, chunk and link every file. Writes to `metadata_store`
    without committing; returns (graph, chunks, files_walked)."""
    metadata_store.clear_repo(repo_id, commit=False)  # replace, don't accumulate, on re-ingest
    graph = DependencyGraph()

    report(f"Walking repo: {repo_path}")
    file_paths = walk_repo(repo_path)
    report(f"Found {len(file_paths)} files")
    python_roots = find_python_source_roots(repo_path)
    ts_config = load_ts_config(repo_path)

    all_chunks = []
    for file_path in file_paths:
        content = load_file(file_path)
        if content is None:
            continue
        rel_path = os.path.relpath(file_path, repo_path)
        language = detect_language(file_path)
        graph.add_file(rel_path)

        analysis = analyze_file(content, rel_path, language)
        metadata_store.add_symbols(repo_id, analysis.symbols)
        metadata_store.add_references(repo_id, rel_path, analysis.references)

        chunks = chunk_file(
            content, rel_path, language, symbols=analysis.symbols, imports=analysis.imports
        )
        metadata_store.add_chunks(chunks, repo_id)
        all_chunks.extend(chunks)

        for imp in analysis.imports:
            targets: list[str] = []
            if language == "python":
                targets = resolve_python_import(
                    imp.imported_module, file_path, repo_path,
                    names=imp.names, source_roots=python_roots,
                )
            elif language in ("typescript", "javascript"):
                hit = resolve_ts_import(imp.imported_module, file_path, repo_path, ts_config)
                targets = [hit] if hit else []
            for target in targets:
                rel_target = os.path.relpath(target, repo_path)
                if rel_target == rel_path:
                    continue  # e.g. `from . import x` inside __init__.py where x is an attribute
                graph.add_import_edge(rel_path, rel_target)
                metadata_store.add_edge(repo_id, rel_path, rel_target, "import")

    metadata_store.prune_references(repo_id)
    return graph, all_chunks, len(file_paths)
