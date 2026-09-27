"""Diff-aware impact: which *symbols* did a change touch, and who uses them?

File-level impact treats every importer of a file as equally affected. A diff
says more: if a commit only changed `HTTPAdapter.cert_verify`, the files that
call `cert_verify` are the ones to look at first.
"""
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from app.models.schemas import ChangedSymbol, DiffImpactResponse

if TYPE_CHECKING:
    from app.core.graph import DependencyGraph
    from app.core.history import CoChange
    from app.storage.faiss_store import FAISSStore
    from app.storage.metadata_store import MetadataStore

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

# A file that uses a changed symbol ranks just under a test named after the
# changed file (0.97) and above plain direct importers (0.95).
_SYMBOL_USE_CONFIDENCE = 0.96


@dataclass
class FileChange:
    path: str
    new_ranges: list[tuple[int, int]] = field(default_factory=list)  # lines in the new version
    old_ranges: list[tuple[int, int]] = field(default_factory=list)  # lines in the old version
    deleted: bool = False


def _strip_prefix(path: str) -> str:
    path = path.strip().split("\t")[0]
    if path.startswith(("a/", "b/")):
        path = path[2:]
    return path


def parse_unified_diff(diff: str) -> list[FileChange]:
    """Changed line ranges per file from `git diff` / `git show` output.

    Pure deletions are recorded on the new side as the line the hunk points
    at, so the enclosing symbol in the new version is still found.
    """
    files: list[FileChange] = []
    current: Optional[FileChange] = None
    old_path = None
    for line in diff.splitlines():
        if line.startswith("--- "):
            old_path = _strip_prefix(line[4:])
        elif line.startswith("+++ "):
            new_path = _strip_prefix(line[4:])
            deleted = new_path == "/dev/null"
            path = old_path if deleted else new_path
            current = FileChange(path=path, deleted=deleted)
            files.append(current)
        elif line.startswith("@@") and current is not None:
            m = _HUNK_RE.match(line)
            if not m:
                continue
            old_start, old_len = int(m.group(1)), int(m.group(2) or 1)
            new_start, new_len = int(m.group(3)), int(m.group(4) or 1)
            if old_len:
                current.old_ranges.append((old_start, old_start + old_len - 1))
            if new_len:
                current.new_ranges.append((new_start, new_start + new_len - 1))
            elif not current.deleted:
                point = max(new_start, 1)
                current.new_ranges.append((point, point))
    return files


def symbols_touched(symbol_rows: list[dict], ranges: list[tuple[int, int]]) -> list[dict]:
    """Innermost symbols overlapping any changed range: a change inside a method
    reports the method, not also its class; a change to class-level lines
    (attributes, docstring) reports the class."""
    touched: dict[tuple, dict] = {}
    for start, end in ranges:
        hits = [
            s for s in symbol_rows
            if s.get("end_line") and s["start_line"] <= end and s["end_line"] >= start
        ]
        for s in hits:
            contains_other = any(
                o is not s and s["start_line"] <= o["start_line"] and o["end_line"] <= s["end_line"]
                and (o["start_line"], o["end_line"]) != (s["start_line"], s["end_line"])
                for o in hits
            )
            if not contains_other:
                touched[(s["qualified_name"], s["start_line"])] = s
    return sorted(touched.values(), key=lambda s: s["start_line"])


def analyze_symbol_changes(
    changes: dict[str, list[str]],
    repo_id: str,
    graph: "DependencyGraph",
    faiss_store: "FAISSStore",
    metadata_store: "MetadataStore",
    embeddings_module,
    depth: int = 3,
    cochange: Optional["CoChange"] = None,
) -> DiffImpactResponse:
    """Impact of changing specific symbols in specific files.

    `changes` maps file → qualified names of the symbols changed in it (empty
    list: module-level change, or unknown). Starts from file-level batch impact
    and ranks files that *use* a changed symbol first. A usage only counts in
    files that depend on the changed file (within `depth` import hops) or are
    the file itself, so an unrelated class's `send` doesn't match.
    """
    from app.core.impact import analyze_impact_batch
    from app.models.schemas import BatchImpactedFile
    from app.core.definitions import is_test_path

    files = sorted(changes)
    base = analyze_impact_batch(
        files, repo_id, graph, faiss_store, metadata_store, embeddings_module,
        depth=depth, cochange=cochange,
    )
    merged: dict[str, BatchImpactedFile] = {
        f.file_path: f for f in base.high_confidence + base.medium_confidence + base.related
    }
    changed_symbols: list[ChangedSymbol] = []

    for file_path in files:
        dependents = {d["file"] for d in graph.dependents_of(file_path, depth=depth)}
        for qualified in changes[file_path]:
            # Matched by name within dependents. Generic method names (`read`,
            # `get`) over-match, but also requiring the class name in the user
            # file lost every gain on the history eval: methods are mostly
            # called on instances obtained elsewhere (`r.connection.send(...)`).
            bare = qualified.rsplit(".", 1)[-1]
            users = sorted({
                r["file_path"] for r in metadata_store.find_references(repo_id, bare)
                if r["file_path"] in dependents
            })
            changed_symbols.append(ChangedSymbol(file_path=file_path, qualified_name=qualified, used_in=users))
            for user in users:
                if user in changes:
                    continue  # changed in this same diff
                reason = f"uses changed {qualified}"
                existing = merged.get(user)
                if existing is None or existing.confidence < _SYMBOL_USE_CONFIDENCE:
                    merged[user] = BatchImpactedFile(
                        file_path=user, reason=reason, confidence=_SYMBOL_USE_CONFIDENCE,
                        depth=existing.depth if existing else 1,
                        triggered_by=sorted(set(existing.triggered_by if existing else []) | {file_path}),
                    )
                else:
                    existing.triggered_by = sorted(set(existing.triggered_by) | {file_path})

    from app.core.impact import _rank_key
    ordered = sorted(merged.values(), key=lambda f: _rank_key(f.file_path, f.confidence, f.depth, cochange))
    high = [f for f in ordered if f.confidence >= 0.7]
    medium = [f for f in ordered if 0.4 <= f.confidence < 0.7]
    related = [f for f in ordered if f.confidence < 0.4]
    return DiffImpactResponse(
        targets=files,
        changed_symbols=changed_symbols,
        high_confidence=high,
        medium_confidence=medium,
        related=related,
        tests=[f for f in ordered if is_test_path(f.file_path)],
    )


def analyze_diff(
    diff: str,
    repo_id: str,
    graph: "DependencyGraph",
    faiss_store: "FAISSStore",
    metadata_store: "MetadataStore",
    embeddings_module,
    depth: int = 3,
    cochange: Optional["CoChange"] = None,
) -> DiffImpactResponse:
    """Impact of a unified diff against the indexed code (e.g. `git diff`).

    Line numbers are matched against the index, so the diff's new side should
    be the code that was ingested, e.g. ingest the working tree and pass
    `git diff HEAD` output. Files not in the index are reported in
    `unindexed_files` and skipped.
    """
    changes: dict[str, list[str]] = {}
    unindexed: list[str] = []
    for fc in parse_unified_diff(diff):
        if fc.path not in graph.G.nodes:
            unindexed.append(fc.path)
            continue
        rows = metadata_store.symbols_in_file(repo_id, fc.path)
        touched = symbols_touched(rows, fc.new_ranges)
        changes[fc.path] = [s["qualified_name"] for s in touched]
    if not changes:
        return DiffImpactResponse(targets=[], unindexed_files=unindexed)
    response = analyze_symbol_changes(
        changes, repo_id, graph, faiss_store, metadata_store, embeddings_module, depth, cochange,
    )
    response.unindexed_files = unindexed
    return response
