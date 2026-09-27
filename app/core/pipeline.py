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
    metadata_store.clear_repo(repo_id)  # replace, don't accumulate, on re-ingest
    graph = DependencyGraph()

    _report(f"Walking repo: {resolved_repo_path}")
    file_paths = walk_repo(resolved_repo_path)
    _report(f"Found {len(file_paths)} files")
    python_roots = find_python_source_roots(resolved_repo_path)
    ts_config = load_ts_config(resolved_repo_path)

    all_chunks = []
    total_symbols = 0

    for file_path in file_paths:
        content = load_file(file_path)
        if content is None:
            continue
        rel_path = os.path.relpath(file_path, resolved_repo_path)
        language = detect_language(file_path)
        graph.add_file(rel_path)

        analysis = analyze_file(content, rel_path, language)
        symbols, imports = analysis.symbols, analysis.imports

        for sym in symbols:
            metadata_store.upsert_symbol(
                name=sym.name, repo_id=repo_id, file_path=rel_path,
                line=sym.start_line, kind=sym.kind,
            )
            total_symbols += 1

        chunks = chunk_file(content, rel_path, language)
        for chunk in chunks:
            metadata_store.upsert_chunk(chunk, repo_id)
        all_chunks.extend(chunks)

        for imp in imports:
            targets: list[str] = []
            if language == "python":
                targets = resolve_python_import(
                    imp.imported_module, file_path, resolved_repo_path,
                    names=imp.names, source_roots=python_roots,
                )
            elif language in ("typescript", "javascript"):
                hit = resolve_ts_import(imp.imported_module, file_path, resolved_repo_path, ts_config)
                targets = [hit] if hit else []
            for target in targets:
                rel_target = os.path.relpath(target, resolved_repo_path)
                if rel_target == rel_path:
                    continue  # e.g. `from . import x` inside __init__.py where x is an attribute
                graph.add_import_edge(rel_path, rel_target)
                metadata_store.upsert_edge(repo_id, rel_path, rel_target, "import")

    graph.save(str(paths.graph_path(repo_id)))
    edge_count = graph.edge_count

    backend = os.environ.get("EMBEDDING_BACKEND", "local").lower()
    model_name = get_embedding_model_name(backend)

    if all_chunks:
        _report(f"Embedding {len(all_chunks)} chunks...")
        texts = [c.content for c in all_chunks]
        chunk_ids = [c.chunk_id for c in all_chunks]
        embeddings = embed_texts(texts, backend=backend)
        dim = embeddings.shape[1]
        faiss_store = FAISSStore(dim=dim, embedding_backend=backend, embedding_model=model_name)
        faiss_store.add(embeddings, chunk_ids)
    else:
        # Still write an (empty) index so a repo with zero indexable chunks
        # doesn't leave get_repo_state() unable to find anything to load.
        faiss_store = FAISSStore(
            dim=get_embedding_dim(backend), embedding_backend=backend, embedding_model=model_name
        )
    faiss_store.save(str(paths.index_path(repo_id)))

    summary = {
        "repo_id": repo_id,
        "files_indexed": len(file_paths),
        "chunks_indexed": len(all_chunks),
        "symbols_extracted": total_symbols,
        "edges_in_graph": edge_count,
        "embedding_backend": backend,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(paths.meta_path(repo_id), "w") as f:
        json.dump(summary, f)

    return summary
