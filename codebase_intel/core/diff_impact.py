"""Diff-aware impact: which *symbols* did a change touch, and who uses them?

File-level impact treats every importer of a file as equally affected. A diff
says more: if a commit only changed `HTTPAdapter.cert_verify`, the files that
call `cert_verify` are the ones to look at first.
"""
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from codebase_intel.core.usages import symbol_users
from codebase_intel.models.schemas import ChangedSymbol, DiffImpactResponse

if TYPE_CHECKING:
    from codebase_intel.core.graph import DependencyGraph
    from codebase_intel.core.history import CoChange
    from codebase_intel.storage.metadata_store import MetadataStore

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


def _ranges(lines: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for n in sorted(set(lines)):
        if out and n == out[-1][1] + 1:
            out[-1] = (out[-1][0], n)
        else:
            out.append((n, n))
    return out


def parse_unified_diff(diff: str) -> list[FileChange]:
    """Changed lines per file from `git diff` / `git show` output.

    Only lines actually removed (old side) or added (new side) count, never
    the surrounding context. A pure insertion is recorded on the old side as
    the line before it, and a pure deletion likewise on the new side, so the
    enclosing symbol is found whichever version the index holds.
    """
    files: list[FileChange] = []
    current: Optional[FileChange] = None
    old_path = None
    old_lines: list[int] = []
    new_lines: list[int] = []
    left_old = left_new = 0         # body lines remaining in the current hunk
    old_no = new_no = 0             # line numbers of the next body line
    block_old: list[int] = []       # the current run of -/+ lines
    block_new: list[int] = []
    block_before = (0, 0)           # (old, new) line just before the run
    added = False                   # the current file is new (no old side)

    def end_block() -> None:
        nonlocal block_old, block_new
        if block_old or block_new:
            if not added:
                old_lines.extend(block_old or [max(block_before[0], 1)])
            if not (current and current.deleted):
                new_lines.extend(block_new or [max(block_before[1], 1)])
        block_old, block_new = [], []

    def end_file() -> None:
        end_block()
        if current is not None:
            current.old_ranges = _ranges(old_lines)
            current.new_ranges = _ranges(new_lines)

    for line in diff.splitlines():
        if left_old > 0 or left_new > 0:
            tag = line[:1]
            if tag == "-":
                if not block_old and not block_new:
                    block_before = (old_no - 1, new_no - 1)
                block_old.append(old_no)
                old_no += 1
                left_old -= 1
            elif tag == "+":
                if not block_old and not block_new:
                    block_before = (old_no - 1, new_no - 1)
                block_new.append(new_no)
                new_no += 1
                left_new -= 1
            elif tag == "\\":
                pass  # "\ No newline at end of file"
            else:  # context
                end_block()
                old_no += 1
                new_no += 1
                left_old -= 1
                left_new -= 1
            continue
        if line.startswith("--- "):
            end_file()
            old_path = _strip_prefix(line[4:])
            current = None
        elif line.startswith("+++ "):
            new_path = _strip_prefix(line[4:])
            deleted = new_path == "/dev/null"
            current = FileChange(path=old_path if deleted else new_path, deleted=deleted)
            files.append(current)
            old_lines, new_lines = [], []
            added = old_path == "/dev/null"
        elif line.startswith("@@") and current is not None:
            m = _HUNK_RE.match(line)
            if not m:
                continue
            end_block()
            old_no, left_old = int(m.group(1)), int(m.group(2) or 1)
            new_no, left_new = int(m.group(3)), int(m.group(4) or 1)
            # An empty range's start is the line *before* it (`+41,0`: deleted after line 41)
            old_no += left_old == 0
            new_no += left_new == 0
    end_file()
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
    metadata_store: "MetadataStore",
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
    from codebase_intel.core.impact import analyze_impact_batch, combine_evidence
    from codebase_intel.models.schemas import BatchImpactedFile
    from codebase_intel.core.definitions import is_test_path

    files = sorted(changes)
    base = analyze_impact_batch(
        files, repo_id, graph, metadata_store,
        depth=depth, cochange=cochange,
    )
    merged: dict[str, BatchImpactedFile] = {
        f.file_path: f for f in base.high_confidence + base.medium_confidence + base.related
    }
    changed_symbols: list[ChangedSymbol] = []

    for file_path in files:
        for qualified in changes[file_path]:
            users = symbol_users(repo_id, qualified, file_path, graph, metadata_store, depth)
            changed_symbols.append(ChangedSymbol(file_path=file_path, qualified_name=qualified, used_in=users))
            for user in users:
                if user in changes:
                    continue  # changed in this same diff
                reason = f"uses changed {qualified}"
                existing = merged.get(user)
                if existing is None:
                    merged[user] = BatchImpactedFile(
                        file_path=user, reason=reason, confidence=_SYMBOL_USE_CONFIDENCE,
                        depth=1, triggered_by=[file_path],
                    )
                elif not existing.reason.startswith("uses changed "):
                    # Combined with the file-level evidence as in combine_evidence
                    existing.confidence, existing.reason, _ = combine_evidence([
                        (_SYMBOL_USE_CONFIDENCE, reason, 0), (existing.confidence, existing.reason, 0)])
                    existing.triggered_by = sorted(set(existing.triggered_by) | {file_path})
                else:
                    existing.triggered_by = sorted(set(existing.triggered_by) | {file_path})

    from codebase_intel.core.impact import _rank_key
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
    metadata_store: "MetadataStore",
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
        changes, repo_id, graph, metadata_store, depth, cochange,
    )
    response.unindexed_files = unindexed
    return response
