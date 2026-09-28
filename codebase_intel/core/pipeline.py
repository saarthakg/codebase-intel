"""Indexing: parse the repo, link imports, infer method receivers and mine
git co-change history. Shared by the CLI, the MCP server and the evals."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from codebase_intel.core import paths
from codebase_intel.core.graph import (
    DependencyGraph,
    find_python_source_roots,
    load_ts_config,
    resolve_python_import,
    resolve_ts_import,
)
from codebase_intel.core.history import cochange_for_repo
from codebase_intel.core.ingest import RepoScan, detect_language, load_file, scan_repo
from codebase_intel.core.symbols import analyze_file, python_parser
from codebase_intel.core.validation import validate_repo_id
from codebase_intel.storage.metadata_store import MetadataStore

ProgressFn = Optional[Callable[[str], None]]


class IngestError(ValueError):
    """Raised for user-fixable indexing problems (bad path, bad repo_id, etc.)."""


def run_ingestion(
    repo_path: str, repo_id: str, progress: ProgressFn = None,
    history_path: Optional[str] = None, extra_meta: Optional[dict] = None,
) -> dict:
    """Index `repo_path` under `repo_id`. Returns a dict of summary counts.

    Each run fully replaces the previous one for this repo_id. Git history is
    read from `history_path` (default `repo_path`): a snapshot of a commit
    has the files but not the history.
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
    try:
        graph, scan = _index_files(resolved_repo_path, repo_id, metadata_store, _report,
                                   history_path or resolved_repo_path)
        # Commit only once everything has succeeded: an error mid-index leaves
        # the previous index intact instead of a half-cleared database.
        metadata_store.commit()
    except BaseException:
        metadata_store.rollback()
        raise
    total_symbols = metadata_store.count_symbols(repo_id)
    total_references = metadata_store.count_references(repo_id)
    history_files = len(metadata_store.load_cochange(repo_id).file_commits)
    metadata_store.close()

    graph.save(str(paths.graph_path(repo_id)))
    paths.legacy_graph_path(repo_id).unlink(missing_ok=True)  # superseded by the JSON graph

    summary = {
        "repo_id": repo_id,
        "repo_path": resolved_repo_path,
        "files_indexed": scan.indexed,
        "files_skipped": dict(scan.skipped),
        "symbols_extracted": total_symbols,
        "references_indexed": total_references,
        "files_with_history": history_files,
        "edges_in_graph": graph.edge_count,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        **(extra_meta or {}),
    }
    with open(paths.meta_path(repo_id), "w") as f:
        json.dump(summary, f)
    return summary


def _index_method_refs(repo_id: str, sources: dict[str, str], metadata_store: MetadataStore) -> None:
    """Two passes over the Python files: collect class/return/attribute facts
    repo-wide, then infer each method reference's receiver type."""
    from codebase_intel.core.typeinfer import TypeIndex, attribute_refs, collect_facts
    parser = python_parser()
    if parser is None or not sources:
        return
    trees = {rel: (parser.parse(src.encode("utf-8")), src.encode("utf-8")) for rel, src in sources.items()}
    facts = [collect_facts(tree.root_node, raw, rel) for rel, (tree, raw) in trees.items()]
    index = TypeIndex.build(facts)
    method_names = {m for cs in index.classes.values() for c in cs for m in c.methods}
    metadata_store.add_class_bases(
        repo_id, [(c.name, b) for ff in facts for c in ff.classes for b in c.bases]
    )
    for rel, (tree, raw) in trees.items():
        metadata_store.add_method_refs(repo_id, rel, attribute_refs(tree.root_node, raw, rel, index, method_names))


def _index_files(
    repo_path: str, repo_id: str, metadata_store: MetadataStore, report: Callable[[str], None],
    history_path: str,
) -> tuple[DependencyGraph, "RepoScan"]:
    """Walk, parse and link every file. Writes to `metadata_store` without
    committing; returns (graph, scan)."""
    metadata_store.clear_repo(repo_id, commit=False)  # replace, don't accumulate, on re-ingest
    graph = DependencyGraph()

    report(f"Walking repo: {repo_path}")
    scan = scan_repo(repo_path)
    file_paths = scan.files
    skipped = ", ".join(f"{n} {reason.replace('_', ' ')}" for reason, n in sorted(scan.skipped.items()))
    report(f"Found {len(file_paths)} files" + (" (.gitignore applied)" if scan.used_git else "")
           + (f"; skipped {skipped}" if skipped else ""))
    python_roots = find_python_source_roots(repo_path)
    ts_config = load_ts_config(repo_path)

    scanned = {os.path.relpath(f, repo_path) for f in file_paths}
    indexed = 0
    python_sources: dict[str, str] = {}  # rel path → content, for typed method refs
    for file_path in file_paths:
        content = load_file(file_path)
        if content is None:
            continue
        rel_path = os.path.relpath(file_path, repo_path)
        language = detect_language(file_path)
        graph.add_file(rel_path)
        indexed += 1

        analysis = analyze_file(content, rel_path, language)
        if language == "python":
            python_sources[rel_path] = content
        metadata_store.add_symbols(repo_id, analysis.symbols)
        metadata_store.add_references(repo_id, rel_path, analysis.references)
        metadata_store.add_files(repo_id, [(rel_path, language)])

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
                if rel_target not in scanned:
                    continue  # resolves to a file that isn't indexed (e.g. skipped as too large)
                graph.add_import_edge(rel_path, rel_target)
                metadata_store.add_edge(repo_id, rel_path, rel_target, "import")

    metadata_store.prune_references(repo_id)
    _index_method_refs(repo_id, python_sources, metadata_store)

    cochange = cochange_for_repo(history_path, keep=set(graph.G.nodes))
    metadata_store.save_cochange(repo_id, cochange)
    if cochange.commits_used:
        report(f"Co-change history: {cochange.commits_used} commits")
    unreadable = len(file_paths) - indexed
    if unreadable:
        scan.skipped["binary_or_unreadable"] += unreadable
    scan.indexed = indexed
    return graph, scan
