"""Which lines, and which symbols, a unified diff touches."""
import re
from dataclasses import dataclass, field
from typing import Optional

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


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
