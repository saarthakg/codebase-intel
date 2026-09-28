"""Undone work: this turn removed lines that earlier work put there.

"Earlier work" is what existed when the current prompt was submitted and
wasn't there before the session: the agent's fixes from earlier turns, and
edits the user made between turns. The user's own uncommitted edits from
before the session count too. Lines that were only moved (the same text added
elsewhere in this turn) and trivial lines (blank, comments, short tokens like
`else:`) are ignored. Snapshots, not git blame: most of this work was never
committed.

Removing earlier work can be right, so this is fix-or-justify, never a block.
"""
import hashlib
from collections import Counter

from notyet import snapshot
from notyet.findings import Context, EngineResult, Finding

MIN_LINE = 8   # shorter stripped lines (`return`, `else:`, `})`) say nothing about intent


def run(ctx: Context) -> EngineResult:
    result = EngineResult()
    s = ctx.session
    turn_start = s.turns[-1]["tree"] if s.turns else ctx.baseline_tree
    if turn_start == ctx.current_tree:
        return result

    protected: dict[str, Counter] = {}
    origin: dict[str, str] = {}
    if s.baseline_head and s.baseline_source == "session-start" and s.baseline_head != ctx.baseline_tree:
        _add(protected, origin, _lines(ctx.root, s.baseline_head, ctx.baseline_tree, "+"),
             "your uncommitted edits from before the session")
    if turn_start != ctx.baseline_tree:
        _add(protected, origin, _lines(ctx.root, ctx.baseline_tree, turn_start, "+"), "an earlier turn of this session")
    if not protected:
        return result

    removed = _lines(ctx.root, turn_start, ctx.current_tree, "-")
    moved = Counter(text for lines in _lines(ctx.root, turn_start, ctx.current_tree, "+").values() for text in lines)
    for path in sorted(removed):
        keep = protected.get(path)
        if not keep:
            continue
        undone = []
        for text in removed[path]:
            if keep[text] > 0 and moved[text] <= 0:
                keep[text] -= 1
                undone.append(text)
            elif moved[text] > 0:
                moved[text] -= 1
        if not undone:
            continue
        digest = hashlib.sha1("\n".join(undone).encode()).hexdigest()[:10]
        result.findings.append(Finding(
            rule="undone-work", severity="resolve", location=path,
            title=f"{len(undone)} line(s) in {path} from {origin[path]} were removed in this turn",
            evidence=[t[:120] for t in undone[:3]], key=f"{path}|{digest}",
            action="Restore them if removing them wasn't part of the request; otherwise acknowledge why."))
    return result


def _add(protected: dict[str, Counter], origin: dict[str, str], lines: dict[str, list[str]], label: str) -> None:
    for path, texts in lines.items():
        protected.setdefault(path, Counter()).update(texts)
        origin.setdefault(path, label)


def _lines(root: str, old: str, new: str, sign: str) -> dict[str, list[str]]:
    """Non-trivial added ("+") or removed ("-") lines per file between two trees."""
    diff = snapshot.git(root, "diff", "-U0", "--no-color", "--no-ext-diff", "--no-renames", old, new)
    out: dict[str, list[str]] = {}
    path = None
    header = "+++ " if sign == "+" else "--- "
    prefix = "+++ b/" if sign == "+" else "--- a/"
    for line in diff.splitlines():
        if line.startswith(header):
            path = line[len(prefix):] if line.startswith(prefix) else None
        elif line.startswith(("+++ ", "--- ")):
            continue
        elif path and line.startswith(sign):
            text = line[1:].strip()
            if len(text) >= MIN_LINE and not text.startswith(("#", "//")):
                out.setdefault(path, []).append(text)
    return out
